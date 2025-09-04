"""ChatGPT Plugiun Tool."""

from typing import List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.openapi.base import OpenAPIToolSpec


class ChatGPTPluginToolSpec(BaseToolSpec):
    """
    ChatGPT Plugin Tool.

    This tool leverages the OpenAPI tool spec to automatically load ChatGPT
    plugins from a manifest file.
    You should also provide the Requests tool spec to allow the Agent to make calls to the OpenAPI endpoints
    To use endpoints with authorization, use the Requests tool spec with the authorization headers
    """

    spec_functions = ["load_openapi_spec", "describe_plugin"]

    def __init__(
        self, manifest: Optional[dict] = None, manifest_url: Optional[str] = None
    ):
        import yaml

        if manifest and manifest_url:
            raise ValueError("You cannot provide both a manifest and a manifest_url")
        elif manifest:
            pass
        elif manifest_url:
            response = requests.get(manifest_url).text
            manifest = yaml.safe_load(response)
        else:
            raise ValueError("You must provide either a manifest or a manifest_url")

        if manifest["api"]["type"] != "openapi":
            raise ValueError(
                f'API type must be "openapi", not "{manifest["api"]["type"]}"'
            )

        if manifest["auth"]["type"] != "none":
            raise ValueError("Authentication cannot be supported for ChatGPT plugins")

        self.openapi = OpenAPIToolSpec(url=manifest["api"]["url"])

        self.plugin_description = f"""
            'human_description': {manifest["description_for_human"]}
            'model_description': {manifest["description_for_model"]}
        """

    def load_openapi_spec(self) -> List[Document]:
        """
        You are an AI agent specifically designed to retrieve information by making web requests to an API based on an OpenAPI specification.

        Here's a step-by-step guide to assist you in answering questions:

        1. Determine the base URL required for making the request

        2. Identify the relevant paths necessary to address the question

        3. Find the required parameters for making the request

        4. Perform the necessary requests to obtain the answer

        Returns:
            Document: A List of Document objects describing the OpenAPI spec

        """
        return self.openapi.load_openapi_spec()

    def describe_plugin(self) -> List[Document]:
        return self.plugin_description
