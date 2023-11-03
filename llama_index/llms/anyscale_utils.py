from typing import Any, Dict, List, Sequence

from llama_index.llms.base import ChatMessage, MessageRole

LLAMA_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
}

MISTRAL_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.1": 4096,
}

ALL_AVAILABLE_MODELS = {
    **LLAMA_MODELS,
    **MISTRAL_MODELS,
}

DISCONTINUED_MODELS: Dict[str, int] = {}


def anyscale_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = anyscale_modelname_to_contextsize(model_name)
    """
    # handling finetuned models
    # TO BE FILLED

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"Anyscale hosted model {modelname} has been discontinued. "
            "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anyscale model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def _message_to_anyscale_prompt(message: ChatMessage) -> Dict[str, Any]:
    if message.role == MessageRole.USER:
        prompt = {"role": "user", "content": message.content}
    elif message.role == MessageRole.ASSISTANT:
        prompt = {"role": "assistant", "content": message.content}
    elif message.role == MessageRole.SYSTEM:
        prompt = {"role": "system", "content": message.content}
    elif message.role == MessageRole.FUNCTION:
        raise ValueError(f"Message role {MessageRole.FUNCTION} is not supported.")
    else:
        raise ValueError(f"Unknown message role: {message.role}")

    return prompt


def messages_to_anyscale_prompt(messages: Sequence[ChatMessage]) -> List[Dict]:
    if len(messages) == 0:
        raise ValueError("Got empty list of messages.")

    return [_message_to_anyscale_prompt(message) for message in messages]
