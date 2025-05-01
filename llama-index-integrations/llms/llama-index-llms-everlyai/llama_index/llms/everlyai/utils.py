from typing import Dict

LLAMA_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf": 4096,
}

ALL_AVAILABLE_MODELS = {
    **LLAMA_MODELS,
}

DISCONTINUED_MODELS: Dict[str, int] = {}


def everlyai_modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = everlyai_modelname_to_contextsize(model_name)

    """
    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"EverlyAI hosted model {modelname} has been discontinued. "
            "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid EverlyAI model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size
