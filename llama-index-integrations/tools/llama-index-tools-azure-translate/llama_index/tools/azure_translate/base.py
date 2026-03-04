"""Azure Translate tool spec."""

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ENDPOINT_BASE_URL = "https://api.cognitive.microsofttranslator.com/translate"


class AzureTranslateToolSpec(BaseToolSpec):
    """Azure Translate tool spec."""

    spec_functions = ["translate"]

    def __init__(self, api_key: str, region: str) -> None:
        """Initialize with parameters."""
        self.headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Ocp-Apim-Subscription-Region": region,
            "Content-type": "application/json",
        }

    def translate(self, text: str, language: str):
        """
        Use this tool to translate text from one language to another.
        The source language will be automatically detected. You need to specify the target language
        using a two character language code.

        Args:
            language (str): Target translation language.

        """
        request = requests.post(
            ENDPOINT_BASE_URL,
            params={"api-version": "3.0", "to": language},
            headers=self.headers,
            json=[{"text": text}],
        )
        return request.json()
