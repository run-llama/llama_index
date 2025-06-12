import re
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
    "magistral-medium-2506": 40000,
    "magistral-small-2506": 40000,
    "magistral-medium-latest": 40000,
    "magistral-small-latest": 40000,
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
