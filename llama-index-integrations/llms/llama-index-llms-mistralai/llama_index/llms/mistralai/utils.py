from typing import Dict

MISTRALAI_MODELS: Dict[str, int] = {
    "mistral-tiny": 32000,
    "mistral-small": 32000,
    "mistral-medium": 32000,
    "mistral-large": 131000,
    "mistral-saba-latest": 32000,
    "open-mixtral-8x7b": 32000,
    "open-mistral-7b": 32000,
    "open-mixtral-8x22b": 64000,
    "mistral-small-latest": 32000,
    "mistral-medium-latest": 32000,
    "mistral-large-latest": 32000,
    "codestral-latest": 256000,
    "open-mistral-nemo-latest": 131000,
    "ministral-8b-latest": 131000,
    "ministral-3b-latest": 131000,
    "pixtral-large-latest": 131000,
    "pixtral-12b-2409": 131000,
}

MISTRALAI_FUNCTION_CALLING_MODELS = (
    "mistral-large-latest",
    "open-mixtral-8x22b",
    "ministral-8b-latest",
    "ministral-3b-latest",
    "mistral-small-latest",
    "codestral-latest",
    "open-mistral-nemo-latest",
    "pixtral-large-latest",
    "pixtral-12b-2409",
)

MISTRALAI_CODE_MODELS = "codestral-latest"


def mistralai_modelname_to_contextsize(modelname: str) -> int:
    # handling finetuned models
    if modelname.startswith("ft:"):
        modelname = modelname.split(":")[1]

    if modelname not in MISTRALAI_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid MistralAI model name."
            "Known models are: " + ", ".join(MISTRALAI_MODELS.keys())
        )

    return MISTRALAI_MODELS[modelname]


def is_mistralai_function_calling_model(modelname: str) -> bool:
    return modelname in MISTRALAI_FUNCTION_CALLING_MODELS


def is_mistralai_code_model(modelname: str) -> bool:
    return modelname in MISTRALAI_CODE_MODELS
