import os
from typing import Optional, Union

WATSONX_MODELS = {
    "google/flan-t5-xxl": 4096,
    "google/flan-ul2": 4096,
    "bigscience/mt0-xxl": 4096,
    "eleutherai/gpt-neox-20b": 8192,
    "bigcode/starcoder": 8192,
    "meta-llama/llama-2-70b-chat": 4096,
    "ibm/mpt-7b-instruct2": 2048,
    "ibm/granite-13b-instruct-v1": 8192,
    "ibm/granite-13b-chat-v1": 8192,
}


def watsonx_model_to_context_size(model_id: str) -> Union[int, None]:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        model_id: The model name we want to know the context size for.

    Returns:
        The maximum context size
    """
    token_limit = WATSONX_MODELS.get(model_id, None)

    if token_limit is None:
        raise ValueError(f"Model name {model_id} not found in {WATSONX_MODELS.keys()}")

    return token_limit


def get_from_param_or_env_without_error(
    param: Optional[str] = None,
    env_key: Optional[str] = None,
) -> Union[str, None]:
    """Get a value from a param or an environment variable without error."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        return None
