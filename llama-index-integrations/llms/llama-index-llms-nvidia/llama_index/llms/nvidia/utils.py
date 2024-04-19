from typing import Dict

API_CATALOG_MODELS: Dict[str, int] = {
    "mistralai/mistral-7b-instruct-v0.2": 16384,
    "mistralai/mixtral-8x7b-instruct-v0.1": 16384,
    "google/gemma-7b": 4096,
    "google/gemma-2b": 4096,
    "google/codegemma-7b": 4096,
    "meta/codellama-70b": 1024,
    "meta/llama2-70b": 1024,
}


def playground_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in API_CATALOG_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid AI Playground model name."
            "Known models are: " + ", ".join(API_CATALOG_MODELS.keys())
        )

    return API_CATALOG_MODELS[modelname]
