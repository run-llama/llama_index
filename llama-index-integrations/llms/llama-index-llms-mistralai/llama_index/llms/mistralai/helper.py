import requests
from typing import List, Dict, Any


def extract_model_name(model):
    return model["id"]


class MistralHelper:
    def __init__(self, api_key: str) -> None:
        """
        Initialize MistralHelper with API key.

        Args:
            api_key: API key for MistralAI.
        """
        self.api_key = api_key
        self.refresh_models()

    def refresh_models(self) -> None:
        """Refresh the list of available models from MistralAI."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = "https://api.mistral.ai/v1/models"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        model_data: List[Dict[str, Any]] = response.json()["data"]

        self.mistralai_models = {model["id"]: model["max_context_length"] for model in model_data}
        self.function_calling_models = list(
            map(
                extract_model_name,
                filter(lambda m: m.get("capabilities", {}).get("function_calling"), model_data)
            )
        )
        self.coding_models = list(
            map(
                extract_model_name,
                filter(lambda m: m.get("capabilities", {}).get("completion_chat") and
                                "coding" in m.get("description", "").lower(), model_data)
            )
        )
        self.reasoning_models = list(
            map(
                extract_model_name,
                filter(lambda m: m.get("capabilities", {}).get("completion_chat") and
                                "reasoning" in m.get("description", "").lower(), model_data)
            )
        )

    def get_mistralai_models(self) -> Dict[str, int]:
        """
        Get the dictionary of available MistralAI models and their context sizes.

        Returns:
            A dictionary mapping model names to their max_context_length.
        """
        return self.mistralai_models

    def get_function_calling_models(self) -> List[str]:
        """
        Get the list of available MistralAI models that support function calling.

        Returns:
            A list of model names that support function calling.
        """
        return self.function_calling_models

    def get_coding_models(self) -> List[str]:
        """
        Get the list of available MistralAI models that are designed for coding tasks.

        Returns:
            A list of model names that are coding models.
        """
        return self.coding_models


    def get_reasoning_models(self) -> List[str]:
        """
        Get the list of available MistralAI models that are designed for reasoning tasks.

        Returns:
            A list of model names that are reasoning models.
        """
        return self.reasoning_models


    def modelname_to_contextsize(self, modelname: str) -> int:
        """
        Get the context size for a given MistralAI model.

        Args:
            modelname: The name of the MistralAI model

        Returns:
            The context size (max_context_length) for the model

        Raises:
            ValueError: If the model is not found in the available models
        """
        if modelname.startswith("ft:"):
            modelname = modelname.split(":")[1]

        if modelname not in self.mistralai_models:
            raise ValueError(
                f"Unknown model: {modelname}. Please provide a valid MistralAI model name."
                "Known models are: " + ", ".join(self.mistralai_models.keys())
            )

        return self.mistralai_models[modelname]

    def is_function_calling_model(self, modelname: str) -> bool:
        """
        Check if a model supports function calling.

        Args:
            modelname: The name of the MistralAI model

        Returns:
            True if the model supports function calling, False otherwise
        """
        return modelname in self.function_calling_models

    def is_code_model(self, modelname: str) -> bool:
        """
        Check if a model is specifically designed for coding tasks.

        Args:
            modelname: The name of the MistralAI model

        Returns:
            True if the model is a coding model, False otherwise
        """
        return modelname in self.coding_models