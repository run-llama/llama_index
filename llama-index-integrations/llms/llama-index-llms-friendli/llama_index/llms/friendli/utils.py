from typing import Any, Dict, Sequence
from llama_index.core.base.llms.types import ChatMessage


LLAMA_MODELS = {
    "llama-2-13b-chat": 4096,
    "llama-2-70b-chat": 4096,
}

MISTRAL_MODELS = {
    "mistral-7b-instruct-v0-2": 32768,
    "mixtral-8x7b-instruct-v0-1": 32768,
}

GEMMA_MODELS = {
    "gemma-7b-it": 8192,
}

ALL_AVAILABLE_MODELS = {
    **LLAMA_MODELS,
    **MISTRAL_MODELS,
    **GEMMA_MODELS,
}


def friendli_modelname_to_contextsize(modelname: str) -> int:
    """
    Get a context size of a model from its name.

    Args:
        modelname (str): The name of model.

    Returns:
        int: Context size of the model.

    """
    context_size = ALL_AVAILABLE_MODELS.get(modelname)
    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Friendli model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def get_chat_request(messages: Sequence[ChatMessage]) -> Dict[str, Any]:
    """Get messages for the Friendli chat request."""
    return {
        "messages": [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
    }
