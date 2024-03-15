"""OpenAPI Tool."""

from typing import List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class OpenAPIToolSpec(BaseToolSpec):
    """OpenAPI Tool.

    This tool can be used to parse an OpenAPI spec for endpoints and operations
    Use the RequestsToolSpec to automate requests to the openapi server
    """

    spec_functions = ["load_openapi_spec"]

    def __init__(self, spec: Optional[dict] = None, url: Optional[str] = None):
        import yaml

        if spec and url:
            raise ValueError("Only provide one of OpenAPI dict or url")
        elif spec:
            pass
        elif url:
            response = requests.get(url).text
            spec = yaml.safe_load(response)
        else:
            raise ValueError("You must provide a url or OpenAPI spec as a dict")

        parsed_spec = self.process_api_spec(spec)
        self.spec = Document(text=str(parsed_spec))

    def load_openapi_spec(self) -> List[Document]:
        """
        You are an AI agent specifically designed to retrieve information by making web requests to an API based on an OpenAPI specification.

        Here's a step-by-step guide to assist you in answering questions:

        1. Determine the base URL required for making the request

        2. Identify the relevant paths necessary to address the question

        3. Find the required parameters for making the request

        4. Perform the necessary requests to obtain the answer

        Returns:
            Document: A List of Document objects.
        """
        return [self.spec]

    def process_api_spec(self, spec: dict) -> dict:
        """Perform simplification and reduction on an OpenAPI specification.

        The goal is to create a more concise and efficient representation
        for retrieval purposes.
        """

        def reduce_details(details: dict) -> dict:
            reduced = {}
            if details.get("description"):
                reduced["description"] = details.get("description")
            if details.get("parameters"):
                reduced["parameters"] = [
                    param
                    for param in details.get("parameters", [])
                    if param.get("required")
                ]
            if "200" in details["responses"]:
                reduced["responses"] = details["responses"]["200"]
            return reduced

        def dereference_openapi(openapi_doc):
            """Dereferences a Swagger/OpenAPI document by resolving all $ref pointers."""
            try:
                import jsonschema
            except ImportError:
                raise ImportError(
                    "The jsonschema library is required to parse OpenAPI documents. "
                    "Please install it with `pip install jsonschema`."
                )

            resolver = jsonschema.RefResolver.from_schema(openapi_doc)

            def _dereference(obj):
                if isinstance(obj, dict):
                    if "$ref" in obj:
                        with resolver.resolving(obj["$ref"]) as resolved:
                            return _dereference(resolved)
                    return {k: _dereference(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_dereference(item) for item in obj]
                else:
                    return obj

            return _dereference(openapi_doc)

        spec = dereference_openapi(spec)
        endpoints = []
        for route, operations in spec["paths"].items():
            for operation, details in operations.items():
                if operation in ["get", "post", "patch"]:
                    endpoint_name = f"{operation.upper()} {route}"
                    description = details.get("description")
                    endpoints.append(
                        (endpoint_name, description, reduce_details(details))
                    )

        return {
            "servers": spec["servers"],
            "description": spec["info"].get("description"),
            "endpoints": endpoints,
        }
