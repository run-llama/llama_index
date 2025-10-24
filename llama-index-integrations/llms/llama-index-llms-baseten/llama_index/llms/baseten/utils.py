from typing import List
import warnings
from llama_index.core.bridge.pydantic import BaseModel

# https://docs.baseten.co/development/model-apis/overview#supported-models
# Below is the current list of models supported by Baseten model APIs.
# Other dedicated models are also supported, but not listed here.
SUPPORTED_MODEL_SLUGS = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-k2-instruct-0905",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "openai/gpt-oss-120b",
    "ai-org/GLM-4.6",
]


class Model(BaseModel):
    """
    Model information for Baseten models.

    Args:
        id: unique identifier for the model, passed as model parameter for requests
        model_type: API type (defaults to "chat")
        client: client name

    """

    id: str
    model_type: str = "chat"
    client: str = "Baseten"

    def __hash__(self) -> int:
        return hash(self.id)


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


def get_available_models_dynamic(client) -> List[Model]:
    """
    Dynamically fetch available models from Baseten Model APIs.

    Args:
        client: The OpenAI-compatible client instance

    Returns:
        List of Model objects representing available models

    """
    models = []
    try:
        for element in client.models.list().data:
            model = Model(id=element.id)
            models.append(model)

        # Filter out models that might not work properly with chat completions
        # (Currently no exclusions, but this allows for future filtering)
        exclude = set()
        return [model for model in models if model.id not in exclude]

    except Exception as e:
        warnings.warn(
            f"Failed to fetch models dynamically: {e}. Falling back to static list."
        )
        # Fallback to current static list
        return [Model(id=slug) for slug in SUPPORTED_MODEL_SLUGS]


def validate_model_dynamic(client, model_name: str) -> None:
    """
    Validate model against dynamically fetched list from Baseten Model APIs.

    Args:
        client: The OpenAI-compatible client instance
        model_name: The model name to validate

    Raises:
        ValueError: If the model is not available

    """
    try:
        available_models = get_available_models_dynamic(client)
        available_model_ids = [model.id for model in available_models]

        if model_name not in available_model_ids:
            # Try to find partial matches for helpful error messages
            candidates = [
                model_id for model_id in available_model_ids if model_name in model_id
            ]

            if candidates:
                suggestion = f"Did you mean one of: {candidates[:3]}"
            else:
                suggestion = f"Available models: {available_model_ids[:5]}{'...' if len(available_model_ids) > 5 else ''}"

            raise ValueError(
                f"Model '{model_name}' not found in available models. {suggestion}"
            )

    except Exception as e:
        if "not found in available models" in str(e):
            # Re-raise our validation error
            raise
        else:
            # For other errors, fall back to static validation
            warnings.warn(f"Dynamic validation failed: {e}. Using static validation.")
            validate_model_slug(model_name)
