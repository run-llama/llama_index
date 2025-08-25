from typing import List

# https://docs.baseten.co/development/model-apis/overview#supported-models
# Below is the current list of models supported by Baseten model APIs.
# Other dedicated models are also supported, but not listed here.
SUPPORTED_MODEL_SLUGS = [
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
]


def validate_model_slug(model_id: str) -> None:
    """
    Validate that the model_id is a supported model slug for Baseten Model APIs.

    Args:
        model_id: The model ID to validate

    Raises:
        ValueError: If the model_id is not a supported model slug

    """
    if model_id not in SUPPORTED_MODEL_SLUGS:
        raise ValueError(
            f"Model '{model_id}' is not supported by Baseten Model APIs. "
            f"Supported models are: {', '.join(SUPPORTED_MODEL_SLUGS)}"
        )


def is_supported_model_slug(model_id: str) -> bool:
    """
    Check if the model_id is a supported model slug for Baseten Model APIs.

    Args:
        model_id: The model ID to check

    Returns:
        True if the model_id is supported, False otherwise

    """
    return model_id in SUPPORTED_MODEL_SLUGS


def get_supported_models() -> List[str]:
    """
    Get a list of all supported model slugs for Baseten Model APIs.

    Returns:
        List of supported model slugs

    """
    return SUPPORTED_MODEL_SLUGS.copy()
