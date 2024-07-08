from typing import Union, Sequence, List, Tuple

from ai21.models import ChatMessage as J2ChatMessage, RoleType
from ai21.models.chat import ChatMessage as AI21ChatMessage
from llama_index.core.base.llms.types import ChatMessage, MessageRole

JAMBA_MODELS = {
    "jamba-instruct": 256_000,
}

_SYSTEM_ERR_MESSAGE = "System message must be at beginning of message list."


def ai21_model_to_context_size(model: str) -> Union[int, None]:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        model: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    """
    token_limit = JAMBA_MODELS.get(model, None)

    if token_limit is None:
        raise ValueError(f"Model name {model} not found in {JAMBA_MODELS.keys()}")

    return token_limit


def message_to_ai21_j2_message(
    messages: Sequence[ChatMessage],
) -> Tuple[str, List[J2ChatMessage]]:
    system_message = ""
    converted_messages = []  # type: ignore

    for i, message in enumerate(messages):
        if message.role == MessageRole.SYSTEM:
            if i != 0:
                raise ValueError(_SYSTEM_ERR_MESSAGE)
            else:
                system_message = message.content
        else:
            converted_message = J2ChatMessage(
                role=RoleType[message.role.name], text=message.content
            )
            converted_messages.append(converted_message)

    return system_message, converted_messages


def message_to_ai21_message(message: ChatMessage) -> AI21ChatMessage:
    return AI21ChatMessage(role=message.role, content=message.content)
