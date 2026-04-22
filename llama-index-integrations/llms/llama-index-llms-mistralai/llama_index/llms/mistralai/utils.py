import re
from typing import Dict


#
# https://docs.mistral.ai/getting-started/models/models_overview/#api-versioning
MISTRAL_MODELS_TAGGED_LATEST = {
    "magistral-medium-latest": "magistral-medium-2509",
    "magistral-small-latest": "magistral-small-2509",
    "mistral-medium-latest": "mistral-medium-2508",
    "mistral-large-latest": "mistral-medium-2508",  # Note: points to medium, not large
    "pixtral-large-latest": "pixtral-large-2411",
    "ministral-3b-latest": "ministral-3b-2410",
    "ministral-8b-latest": "ministral-8b-2410",
    "mistral-small-latest": "mistral-small-2506",
    "devstral-small-latest": "devstral-small-2507",
    "devstral-medium-latest": "devstral-medium-2507",
    "codestral-latest": "codestral-2508",
}


MISTRALAI_MODELS: Dict[str, int] = {
    "codestral-2501": 256_000,
    "codestral-2508": 256_000,
    "devstral-medium-2507": 128_000,
    "devstral-small-2507": 128_000,
    "magistral-medium-2506": 40_000,
    "magistral-medium-2507": 40_000,
    "magistral-medium-2509": 128_000,
    "magistral-small-2506": 40_000,
    "magistral-small-2507": 40_000,
    "magistral-small-2509": 128_000,
    "ministral-3b-2410": 128_000,
    "ministral-8b-2410": 128_000,
    "mistral-large-2411": 128_000,
    "mistral-medium-2505": 128_000,
    "mistral-medium-2508": 128_000,
    "mistral-small-2407": 32_000,
    "mistral-small-2506": 128_000,
    "open-mistral-nemo": 128_000,
    "pixtral-12b-2409": 131_000,
    "pixtral-large-2411": 12_8000,
}

for tag, reference in MISTRAL_MODELS_TAGGED_LATEST.items():
    MISTRALAI_MODELS[tag] = MISTRALAI_MODELS[reference]


MISTRALAI_FUNCTION_CALLING_MODELS = (
    "mistral-large-latest",
    "ministral-8b-latest",
    "ministral-3b-latest",
    "mistral-small-latest",
    "codestral-latest",
    "open-mistral-nemo-latest",
    "pixtral-large-latest",
    "pixtral-12b-2409",
    "magistral-medium-2506",
    "magistral-small-2506",
    "magistral-medium-latest",
    "magistral-small-latest",
)

MISTRAL_AI_REASONING_MODELS = (
    "magistral-medium-2506",
    "magistral-small-2506",
    "magistral-medium-latest",
    "magistral-small-latest",
)

MISTRALAI_CODE_MODELS = "codestral-latest"

THINKING_REGEX = re.compile(r"^<think>\n(.*?)\n</think>\n")
THINKING_START_REGEX = re.compile(r"^<think>\n")


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
