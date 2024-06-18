import os
from typing import Optional, Union

WATSONX_MODELS = {
    "google/flan-t5-xl": 4096,
    "google/flan-t5-xxl": 4096,
    "google/flan-ul2": 4096,
    "bigscience/mt0-xxl": 4096,
    "bigcode/starcoder": 8192,  # deprecated, removed after 25.04.2024
    "core42/jais-13b-chat": 8192,
    "codellama/codellama-34b-instruct-hf": 16384,
    "meta-llama/llama-2-13b-chat": 4096,
    "meta-llama/llama-2-70b-chat": 4096,
    "meta-llama/llama-3-8b-instruct": 8192,
    "meta-llama/llama-3-70b-instruct": 8192,
    "ibm-mistralai/merlinite-7b": 32768,
    "ibm-mistralai/mixtral-8x7b-instruct-v01-q": 32768,  # deprecated, removed after 23.05.2024
    "ibm/granite-13b-chat-v2": 8192,
    "ibm/granite-13b-instruct-v2": 8192,
    "ibm/granite-20b-multilingual": 8192,
    "ibm/granite-7b-lab": 8192,
    "ibm/granite-8b-japanese": 8192,
    "mistralai/mixtral-8x7b-instruct-v01": 32768,
    "elyza/elyza-japanese-llama-2-7b-instruct": 4096,
    "mncai/llama2-13b-dpo-v7": 4096,
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
