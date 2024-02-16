from typing import Dict

MISTRALAI_MODELS: Dict[str, int] = {
    "mistral-tiny": 32000,
    "mistral-small": 32000,
    "mistral-medium": 32000,
}


def mistralai_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in MISTRALAI_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid MistralAI model name."
            "Known models are: " + ", ".join(MISTRALAI_MODELS.keys())
        )

    return MISTRALAI_MODELS[modelname]
