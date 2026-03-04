"""OpenAPI Tool."""

import json
from collections import OrderedDict
from typing import List, Optional, Callable

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class OpenAPIToolSpec(BaseToolSpec):
    """
    OpenAPI Tool.

    This tool can be used to parse an OpenAPI spec for endpoints and operations
    Use the RequestsToolSpec to automate requests to the openapi server
    """

    spec_functions = ["load_openapi_spec"]

    def __init__(
        self,
        spec: Optional[dict] = None,
        url: Optional[str] = None,
        operation_id_filter: Callable[[str], bool] = None,
    ):
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

        # TODO: if we retrieved spec from URL, the server URL inside the spec may be relative to
        #  the retrieval URL.
        parsed_spec = self.process_api_spec(spec, operation_id_filter)
        self.spec = Document(text=json.dumps(parsed_spec))

    def load_openapi_spec(self) -> List[Document]:
        """
        You are an AI agent specifically designed to retrieve information by making web requests to
        an API based on an OpenAPI specification.

        Here's a step-by-step guide to assist you in answering questions:

        1. Determine the server base URL required for making the request

        2. Identify the relevant endpoint (a HTTP verb plus path template) necessary to address the
        question

        3. Generate the required parameters and/or request body for making the request to the
        endpoint

        4. Perform the necessary requests to obtain the answer

        Returns:
            Document: A List of Document objects that describes the available API.

        """
        return [self.spec]

    def process_api_spec(
        self, spec: dict, operation_id_filter: Callable[[str], bool]
    ) -> dict:
        """
        Perform simplification and reduction on an OpenAPI specification.

        The goal is to create a more concise and efficient representation
        for retrieval purposes.
        """

        def reduce_details(details: dict) -> dict:
            reduced = OrderedDict()
            if details.get("description"):
                reduced["description"] = details.get("description")
            elif details.get("summary"):
                reduced["description"] = details.get("summary")
            if details.get("parameters"):
                reduced["parameters"] = details.get("parameters", [])
            if details.get("requestBody"):
                reduced["requestBody"] = details.get("requestBody")
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
        for path_template, operations in spec["paths"].items():
            for operation, operation_detail in operations.items():
                operation_id = operation_detail.get("operationId")
                if operation_id_filter is None or operation_id_filter(operation_id):
                    if operation in ["get", "post", "patch", "put", "delete"]:
                        # preserve order so the LLM "reads" the description first before all the
                        # schema details
                        details = OrderedDict()
                        details["verb"] = operation.upper()
                        details["path_template"] = path_template
                        details.update(reduce_details(operation_detail))
                        endpoints.append(details)

        return {
            "servers": spec["servers"],
            "description": spec["info"].get("description"),
            "endpoints": endpoints,
        }
