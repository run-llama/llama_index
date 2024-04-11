from typing import Dict, Any

from llama_index.core.base.llms.types import LogProb, CompletionResponse

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


def process_response(response: Any, completion: str) -> CompletionResponse:
    """
    Processes the response from Aleph Alpha API.

    :param response: The response object from Aleph Alpha API.
    :param completion: The completion text.

    :return: A CompletionResponse object.
    """
    log_probs_formatted = []

    if response.completions and hasattr(response.completions[0], "log_probs"):
        log_probs_extracted = response.completions[0].log_probs or []

        for lp_dict in log_probs_extracted:
            if isinstance(lp_dict, dict):
                for token, log_prob in lp_dict.items():
                    log_probs_formatted.append(LogProb(token=token, logprob=log_prob))

    additional_info = extract_additional_info_from_response(response)

    return CompletionResponse(
        text=completion,
        raw=response.to_json(),
        logprobs=[log_probs_formatted],
        additional_kwargs=additional_info,
    )
