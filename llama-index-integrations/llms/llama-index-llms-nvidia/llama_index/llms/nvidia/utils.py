from typing import Dict, Optional

API_CATALOG_MODELS: Dict[str, int] = {
    "mistralai/mistral-7b-instruct-v0.2": 16384,
    "mistralai/mixtral-8x7b-instruct-v0.1": 16384,
    "mistralai/mixtral-8x22b-instruct-v0.1": 32768,
    "mistralai/mistral-large": 16384,
    "google/gemma-7b": 4096,
    "google/gemma-2b": 4096,
    "google/codegemma-7b": 4096,
    "meta/llama2-70b": 1024,
    "meta/codellama-70b": 1024,
    "meta/llama3-8b-instruct": 6000,
    "meta/llama3-70b-instruct": 6000,
}


def catalog_modelname_to_contextsize(modelname: str) -> Optional[int]:
    return API_CATALOG_MODELS.get(modelname, None)
