from typing import Dict

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

LUMINOUS_MODELS: Dict[str, int] = {
    "luminous-base": 2048,
    "luminous-extended": 2048,
    "luminous-supreme": 2048,
    "luminous-base-control": 2048,
    "luminous-extended-control": 2048,
    "luminous-supreme-control": 2048,
}


def alephalpha_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in LUMINOUS_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid AlephAlpha model name."
            "Known models are: " + ", ".join(LUMINOUS_MODELS.keys())
        )

    return LUMINOUS_MODELS[modelname]
