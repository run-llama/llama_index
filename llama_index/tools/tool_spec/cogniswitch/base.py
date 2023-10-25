import requests
from typing import Any, Dict, List, Optional, Type
from llama_index.bridge.pydantic import BaseModel
from llama_index.tools.tool_spec.base import BaseToolSpec


class CogniswitchToolSpec(BaseToolSpec):
    """Cogniswitch Tool Spec.
    A tool used to store data using the Cogniswitch service.
    """

    spec_functions = ["store_data", "query_knowledge"]

    def __init__(self, cs_token: str, OAI_token: str, apiKey: str) -> None:
        self.cs_token = cs_token
        self.OAI_token = OAI_token
        self.apiKey = apiKey

    def store_data(
        self,
        url: Optional[str] = None,
        file: Optional[str] = None,
        document_name: Optional[str] = None,
        document_description: Optional[str] = None,
    ) -> dict:
        """
        Store data using the Cogniswitch service.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            url (Optional[str]): URL link.
            file (Optional[str]): file path of your file.
            the current files supported by the files are
            .txt, .pdf, .docx, .doc, .html
            document_name (Optional[str]): Name of the document you are uploading.
            document_description (Optional[str]): Description of the document.



        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        if not file:
            api_url = (
                "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/url"
            )
            headers = {
                "apiKey": self.apiKey,
                "openAIToken": self.OAI_token,
                "platformToken": self.cs_token,
            }
            files = None
            data = {
                "url": url,
                "documentName": document_name,
                "documentDescription": document_description,
            }
            response = requests.post(
                api_url, headers=headers, verify=False, data=data, files=files
            )

        if not url:
            api_url = (
                "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/file"
            )

            headers = {
                "apiKey": self.apiKey,
                "openAIToken": self.OAI_token,
                "platformToken": self.cs_token,
            }
            if file is not None:
                files = {"file": open(file, "rb")}
            else:
                files = None
            data = {
                "url": url,
                "documentName": document_name,
                "documentDescription": document_description,
            }
            response = requests.post(
                api_url, headers=headers, verify=False, data=data, files=files
            )
        if response.status_code == 200:
            return response.json()
        else:
            # error_message = response.json()["message"]
            return {
                "message": "Bad Request",
            }

    def query_knowledge(self, query: str) -> dict:
        """
        Send a query to the Cogniswitch service and retrieve the response.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            query (str): Query to be answered.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        api_url = "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"

        headers = {
            "apiKey": self.apiKey,
            "platformToken": self.cs_token,
            "openAIToken": self.OAI_token,
        }

        data = {"query": query}
        response = requests.post(api_url, headers=headers, verify=False, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            # error_message = response.json()["message"]
            return {
                "message": "Bad Request",
            }

    def get_fn_schema_from_fn_name(self, fn_name: str) -> Optional[Type[BaseModel]]:
        pass
