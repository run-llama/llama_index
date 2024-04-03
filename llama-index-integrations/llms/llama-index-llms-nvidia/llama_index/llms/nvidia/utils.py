from typing import Dict

API_CATALOG_MODELS: Dict[str, int] = {
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "google/gemma-7b",
    "meta/codellama-70b",
    "meta/llama2-70b",
}

def playground_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in API_CATALOG_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid AI Playground model name."
            "Known models are: " + ", ".join(API_CATALOG_MODELS.keys())
        )

    return API_CATALOG_MODELS[modelname]
