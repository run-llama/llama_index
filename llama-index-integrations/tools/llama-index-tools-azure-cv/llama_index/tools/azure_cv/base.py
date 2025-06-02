"""Azure Cognitive Vision tool spec."""

from typing import List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

CV_URL_TMPL = "https://{resource}.cognitiveservices.azure.com/computervision/imageanalysis:analyze"


class AzureCVToolSpec(BaseToolSpec):
    """Azure Cognitive Vision tool spec."""

    spec_functions = ["process_image"]

    def __init__(
        self,
        resource: str,
        api_key: str,
        language: Optional[str] = "en",
        api_version: Optional[str] = "2023-04-01-preview",
    ) -> None:
        """Initialize with parameters."""
        self.api_key = api_key
        self.cv_url = CV_URL_TMPL.format(resource=resource)
        self.language = language
        self.api_version = api_version

    def process_image(self, url: str, features: List[str]):
        """
        This tool accepts an image url or file and can process and return a variety of text depending on the use case.
        You can use the features argument to configure what text you want returned.

        Args:
            url (str): The url for the image to caption
            features (List[str]): Instructions on how to process the image. Valid keys are tags, objects, read, caption

        """
        response = requests.post(
            f"{self.cv_url}?features={','.join(features)}&language={self.language}&api-version={self.api_version}",
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            json={"url": url},
        )
        response_json = response.json()
        if "read" in features:
            response_json["readResult"] = response_json["readResult"]["content"]

        return response_json
