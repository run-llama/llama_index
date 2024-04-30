import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
)
from llama_index.core.prompts.chat_prompts import TEXT_QA_SYSTEM_PROMPT
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

COMMAND_MODELS = {
    "command-r": 128000,
    "command-r-plus": 128000,
    "command": 4096,
    "command-nightly": 4096,
    "command-light": 4096,
    "command-light-nightly": 4096,
}

GENERATION_MODELS = {"base": 2048, "base-light": 2048}

REPRESENTATION_MODELS = {
    "embed-english-light-v2.0": 512,
    "embed-english-v2.0": 512,
    "embed-multilingual-v2.0": 256,
}

ALL_AVAILABLE_MODELS = {**COMMAND_MODELS, **GENERATION_MODELS, **REPRESENTATION_MODELS}
CHAT_MODELS = {**COMMAND_MODELS}

logger = logging.getLogger(__name__)


# TODO: decide later where this should be moved
class DocumentMessage(ChatMessage):
    role: MessageRole = MessageRole.USER


COHERE_QA_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        TEXT_QA_SYSTEM_PROMPT,
        DocumentMessage(content="{context_str}"),
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]
)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    try:
        import cohere
    except ImportError as e:
        raise ImportError(
            "You must install the `cohere` package to use Cohere."
            "Please `pip install cohere`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(cohere.errors.ServiceUnavailableError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
    client: Any, max_retries: int, chat: bool = False, **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        is_stream = kwargs.pop("stream", False)
        if chat:
            if is_stream:
                return client.chat_stream(**kwargs)
            else:
                print("$" * 33)
                print("MESSAGE")
                print(kwargs.get("message", None))
                print("DOCUMENTS")
                print(kwargs.get("documents", None))
                print("$" * 33)
                return client.chat(**kwargs)
        else:
            if is_stream:
                return client.generate_stream(**kwargs)
            else:
                return client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    aclient: Any,
    max_retries: int,
    chat: bool = False,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        is_stream = kwargs.pop("stream", False)
        if chat:
            if is_stream:
                return await aclient.chat_stream(**kwargs)
            else:
                return await aclient.chat(**kwargs)
        else:
            if is_stream:
                return await aclient.generate_stream(**kwargs)
            else:
                return await aclient.generate(**kwargs)

    return await _completion_with_retry(**kwargs)


def cohere_modelname_to_contextsize(modelname: str) -> int:
    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)
    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Cohere model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_chat_model(model: str) -> bool:
    return model in COMMAND_MODELS


def messages_to_cohere_history(
    messages: Sequence[ChatMessage],
) -> List[Dict[str, Optional[str]]]:
    role_map = {
        "user": "USER",
        "system": "SYSTEM",
        "chatbot": "CHATBOT",
        "assistant": "CHATBOT",
        "model": "SYSTEM",
        "function": "SYSTEM",
        "tool": "SYSTEM",
    }
    return [
        {"role": role_map[message.role], "message": message.content}
        for message in messages
    ]


def message_to_cohere_documents(message: DocumentMessage) -> List[str]:
    """
    Splits out individual documents from `message` in the format expected by Cohere.chat's
    `document` argument.

    NOTE: current implementation is brittle and depends on the formatting of specific retriever.
    I don't yet understand how to control that retriever logic.

    TODO: make document-splitting logic robust to different retrievers
    TODO: handle additional_kwargs from DocumentMessage
    """
    # TODO: move try/except outside
    # TODO: better: look for pattern of k: v values, then parse anew
    try:
        documents = []
        docs = message.content.split("file_path:")
        for doc in docs:
            if doc:
                split_by_separator = doc.split("\n\n")
                source = split_by_separator[0].strip()
                text = "\n\n".join(split_by_separator[1:]).strip()
                documents.append({"source": source, "text": text})
        return documents
    except Exception:
        # Parsing failed. This is likely because 'message' was built from a different retriever
        # Return a default formatting to avoid breaking pipeline
        # TODO: raise warning?
        return [{"text": message.content}]
