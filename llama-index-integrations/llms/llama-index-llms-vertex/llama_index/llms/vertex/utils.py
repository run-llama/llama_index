# utils script

# generation with retry
import logging
from typing import Any, Callable, Optional

import google.api_core
import vertexai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.language_models import ChatMessage as VertexChatMessage
from vertexai.language_models import InputOutputTextPair
from vertexai.generative_models import FunctionDeclaration, Tool

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

CHAT_MODELS = ["chat-bison", "chat-bison-32k", "chat-bison@001"]
TEXT_MODELS = [
    "text-bison",
    "text-bison-32k",
    "text-bison@001",
    "medlm-medium",
    "medlm-large",
]
CODE_MODELS = ["code-bison", "code-bison-32k", "code-bison@001"]
CODE_CHAT_MODELS = ["codechat-bison", "codechat-bison-32k", "codechat-bison@001"]


logger = logging.getLogger(__name__)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
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
            | retry_if_exception_type(google.api_core.exceptions.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def to_gemini_tools(tools) -> Any:
    func_list = []
    for i, tool in enumerate(tools):
        func_name = f"func_{i}"
        func_name = FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
        )
        func_list.append(func_name)
    gemini_tools = Tool(
        function_declarations=func_list,
    )
    return [gemini_tools]


def completion_with_retry(
    client: Any,
    prompt: Optional[Any],
    max_retries: int = 5,
    chat: bool = False,
    stream: bool = False,
    is_gemini: bool = False,
    datastore: Optional[str] = None,
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
            kwargs = dict(kwargs)
            tools = kwargs.pop("tools", None) if "tools" in kwargs else []
            tools = to_gemini_tools(tools) if tools else []
            generation_config = kwargs if kwargs else {}
            if datastore:
                from vertexai.preview.generative_models import Tool, grounding
                tool = Tool.from_retrieval(
                    grounding.Retrieval(grounding.VertexAISearch(datastore=datastore))
                )
                return generation.send_message(
                    prompt, stream=stream, generation_config=generation_config, tools=[tool]
                )
            return generation.send_message(
                prompt,
                stream=stream,
                tools=tools,
                generation_config=generation_config,
            )
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
    datastore: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        if is_gemini:
            history = params["message_history"] if "message_history" in params else []
            generation = client.start_chat(history=history)
            kwargs = dict(kwargs)
            tools = kwargs.pop("tools", None) if "tools" in kwargs else []
            tools = to_gemini_tools(tools) if tools else []
            generation_config = kwargs if kwargs else {}
            if datastore:
                from vertexai.preview.generative_models import Tool, grounding
                tool = Tool.from_retrieval(
                    grounding.Retrieval(grounding.VertexAISearch(datastore=datastore))
                )
                return await generation.send_message_async(
                    prompt, generation_config=generation_config, tools=[tool]
                )
            return await generation.send_message_async(
                prompt,
                tools=tools,
                generation_config=generation_config,
            )
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
    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
    )


def _parse_message(message: ChatMessage, is_gemini: bool) -> Any:
    if is_gemini:
        from llama_index.llms.vertex.gemini_utils import (
            convert_chat_message_to_gemini_content,
        )

        return convert_chat_message_to_gemini_content(message=message, is_history=False)
    else:
        return message.content


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
    vertex_messages, context = [], None
    for i, message in enumerate(history):
        if i == 0 and message.role == MessageRole.SYSTEM:
            if is_gemini:
                raise ValueError("Gemini model doesn't support system messages")
            context = message.content
        elif message.role in (
            MessageRole.MODEL,
            MessageRole.USER,
        ):
            if is_gemini:
                from llama_index.llms.vertex.gemini_utils import (
                    convert_chat_message_to_gemini_content,
                )

                context = message.content
                vertex_messages.append(
                    convert_chat_message_to_gemini_content(
                        message=message, is_history=True
                    )
                )
            else:
                vertex_message = VertexChatMessage(
                    content=message.content,
                    author=(
                        "bot"
                        if message.role in (MessageRole.ASSISTANT, MessageRole.MODEL)
                        else MessageRole.USER
                    ),
                )
                vertex_messages.append(vertex_message)
        else:
            # Roles are consolidated to two roles in merge_neighboring_same_role_messages
            # Other roles will cause error
            raise ValueError(
                f"Unexpected message with role {message.role} at the position {i}."
            )

    if len(vertex_messages) % 2 != 0:
        raise ValueError("total no of messages should be even")

    return {"context": context, "message_history": vertex_messages}


def _parse_examples(examples: Any) -> Any:
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
            if example.role not in (MessageRole.ASSISTANT, MessageRole.MODEL):
                raise ValueError(
                    f"Expected the second message in a part to be from AI, got "
                    f"{type(example)} for the {i}th message."
                )
            pair = InputOutputTextPair(
                input_text=input_text, output_text=example.content
            )
            example_pairs.append(pair)
    return example_pairs


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
