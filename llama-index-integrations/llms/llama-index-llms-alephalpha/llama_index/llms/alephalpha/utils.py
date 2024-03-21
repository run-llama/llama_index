from typing import Dict, Any

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
    """
    Converts an Aleph Alpha model name to the corresponding context size.

    :param modelname: The name of the Aleph Alpha model.
    :return: The context size for the model.
    """
    if modelname not in LUMINOUS_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid AlephAlpha model name."
            "Known models are: " + ", ".join(LUMINOUS_MODELS.keys())
        )

    return LUMINOUS_MODELS[modelname]


def extract_additional_info_from_response(response) -> Dict[str, Any]:
    """
    Extracts additional information from the Aleph Alpha completion response.

    :param response: The response object from Aleph Alpha API.
    :return: A dictionary with extracted information.
    """
    completion = response.completions[0] if response.completions else {}

    additional_info = {
        "model_version": getattr(response, "model_version", None),
        "log_probs": getattr(completion, "log_probs", None),
        "raw_completion": getattr(completion, "raw_completion", None),
        "finish_reason": getattr(completion, "finish_reason", None),
    }

    return {k: v for k, v in additional_info.items() if v is not None}
