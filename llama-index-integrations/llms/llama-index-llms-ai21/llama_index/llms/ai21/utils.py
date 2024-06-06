from typing import Union
from ai21.models.chat import ChatMessage as AI21ChatMessage
from llama_index.core.base.llms.types import ChatMessage

JAMBA_MODELS = {
    "jamba-instruct": 256_000,
}


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


def message_to_ai21_message(message: ChatMessage) -> AI21ChatMessage:
    return AI21ChatMessage(role=message.role, content=message.content)
