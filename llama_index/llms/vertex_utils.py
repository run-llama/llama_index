# utils script
import base64
# generation with retry
import logging
from typing import Any, Callable, Optional, Union, Dict, List

from google.ai.generativelanguage_v1 import Part
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.vision_models import Image

from llama_index.llms.types import MessageRole, ChatMessage

CHAT_MODELS = ["chat-bison", "chat-bison-32k", "chat-bison@001"]
TEXT_MODELS = ["text-bison", "text-bison-32k", "text-bison@001"]
CODE_MODELS = ["code-bison", "code-bison-32k", "code-bison@001"]
CODE_CHAT_MODELS = ["codechat-bison", "codechat-bison-32k", "codechat-bison@001"]

def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


logger = logging.getLogger(__name__)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    import google.api_core

    min_seconds = 4
    max_seconds = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.Aborted)
            | retry_if_exception_type(google.api_core.exceptions.DeadlineExceeded)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
    client: Any,
    prompt: Optional[str],
    max_retries: int = 5,
    chat: bool = False,
    stream: bool = False,
    is_gemini: bool = False,
    params: Any = {},
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        if is_gemini:
            history = params["message_history"] if "message_history" in params else []

            generation = client.start_chat(history=history)
            generation_config = dict(kwargs)
            return generation.send_message(prompt, stream=stream, generation_config=generation_config)
        elif chat:
            generation = client.start_chat(**params)
            if stream:
                return generation.send_message_streaming(prompt, **kwargs)
            else:
                return generation.send_message(prompt, **kwargs)
        else:
            if stream:
                return client.predict_streaming(prompt, **kwargs)
            else:
                return client.predict(prompt, **kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    client: Any,
    prompt: Optional[str],
    max_retries: int = 5,
    chat: bool = False,
    is_gemini: bool = False,
    params: Any = {},
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        if is_gemini:
            history = params["message_history"] if "message_history" in params else []

            generation = client.start_chat(history=history)
            generation_config = dict(kwargs)
            return await generation.send_message_async(prompt, generation_config=generation_config)
        elif chat:
            generation = client.start_chat(**params)
            return await generation.send_message_async(prompt, **kwargs)
        else:
            return await client.predict_async(prompt, **kwargs)

    return await _completion_with_retry(**kwargs)


def init_vertexai(
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> None:
    """Init vertexai.

    Args:
        project: The default GCP project to use when making Vertex API calls.
        location: The default location to use when making API calls.
        credentials: The default custom
            credentials to use when making API calls. If not provided credentials
            will be ascertained from the environment.

    Raises:
        ImportError: If importing vertexai SDK did not succeed.
    """
    try:
        import vertexai
    except ImportError:
        raise (ValueError(f"Please install vertex AI client by following the steps"))

    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
    )


def _convert_gemini_part_to_prompt(part: Union[str, Dict]) -> Part:
    from vertexai.preview.generative_models import Content, Image, Part
    if isinstance(part, str):
        return Part.from_text(part)

    if not isinstance(part, Dict):
        raise ValueError(
            f"Message's content is expected to be a dict, got {type(part)}!"
        )
    if part["type"] == "text":
        return Part.from_text(part["text"])
    elif part["type"] == "image_url":
        path = part["image_url"]["url"]
        if path.startswith("gs://"):
            raise ValueError("Only local image path is supported!")
        elif path.startswith("data:image/jpeg;base64,"):
            image = Image.from_bytes(base64.b64decode(path[23:]))
        else:
            image = Image.load_from_file(path)
    else:
        raise ValueError("Only text and image_url types are supported!")
    return Part.from_image(image)

def _parse_chat_history(history: Any, is_gemini: bool) -> Any:
    """Parse a sequence of messages into history.

    Args:
        history: The list of messages to re-create the history of the chat.

    Returns:
        A parsed chat history.

    Raises:
        ValueError: If a sequence of message has a SystemMessage not at the
        first place.
    """
    from vertexai.language_models import ChatMessage

    vertex_messages, context = [], None
    for i, message in enumerate(history):
        if i == 0 and message.role == MessageRole.SYSTEM:
            if is_gemini:
                raise ValueError(
                    "Gemini model don't support system messages"
                )
        elif message.role == MessageRole.ASSISTANT or message.role == MessageRole.USER:
            if is_gemini:
                from vertexai.preview.generative_models import Content, Image, Part
                raw_content = message.content
                if isinstance(raw_content, str):
                    raw_content = [raw_content]
                parts = [_convert_gemini_part_to_prompt(part) for part in raw_content]
                vertex_message = Content(role="user" if message.role == MessageRole.USER else "model", parts=parts)
                vertex_messages.append(vertex_message)
            else:
                vertex_message = ChatMessage(content=message.content, author="bot" if message.role == MessageRole.ASSISTANT else "user")
                vertex_messages.append(vertex_message)
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )
    if len(vertex_messages) % 2 != 0:
        raise ValueError("total no of messages should be even")

    return {"context": context, "message_history": vertex_messages}


def _parse_examples(examples: Any) -> Any:
    from vertexai.language_models import InputOutputTextPair

    if len(examples) % 2 != 0:
        raise ValueError(
            f"Expect examples to have an even amount of messages, got {len(examples)}."
        )
    example_pairs = []
    input_text = None
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if not example.role == MessageRole.USER:
                raise ValueError(
                    f"Expected the first message in a part to be from user, got "
                    f"{type(example)} for the {i}th message."
                )
            input_text = example.content
        if i % 2 == 1:
            if not example.role == MessageRole.ASSISTANT:
                raise ValueError(
                    f"Expected the second message in a part to be from AI, got "
                    f"{type(example)} for the {i}th message."
                )
            pair = InputOutputTextPair(
                input_text=input_text, output_text=example.content
            )
            example_pairs.append(pair)
    return example_pairs
