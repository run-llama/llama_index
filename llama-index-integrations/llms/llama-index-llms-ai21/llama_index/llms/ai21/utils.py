from typing import Union

COMPLETE_MODELS = {"j2-light": 8191, "j2-mid": 8191, "j2-ultra": 8191}


def ai21_model_to_context_size(model: str) -> Union[int, None]:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        model: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    """
    token_limit = COMPLETE_MODELS.get(model, None)

    if token_limit is None:
        raise ValueError(f"Model name {model} not found in {COMPLETE_MODELS.keys()}")

    return token_limit
