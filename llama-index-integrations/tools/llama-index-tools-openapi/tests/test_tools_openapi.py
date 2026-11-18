import json
from pathlib import Path
import yaml

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.openapi import OpenAPIToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in OpenAPIToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_opid_filter():
    openapi_spec = load_example_spec()
    llamaindex_tool_spec = OpenAPIToolSpec(
        spec=openapi_spec, operation_id_filter=lambda it: it != "findPetsByTags"
    )
    spec_array = llamaindex_tool_spec.load_openapi_spec()
    deserialized = json.loads(spec_array[0].text)
    endpoints: list = deserialized["endpoints"]
    operation = next(
        filter(lambda it: it["path_template"] == "/pet/findByTags", endpoints), None
    )
    assert operation is None


def test_request_body():
    openapi_spec = load_example_spec()
    llamaindex_tool_spec = OpenAPIToolSpec(spec=openapi_spec)
    spec_array = llamaindex_tool_spec.load_openapi_spec()
    deserialized = json.loads(spec_array[0].text)
    endpoints: list = deserialized["endpoints"]
    operation = next(
        filter(
            lambda it: it["path_template"] == "/pet" and it["verb"] == "PUT", endpoints
        )
    )
    assert isinstance(operation["requestBody"], dict)


def test_path_level_parameters_are_not_operations():
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Pets", "version": "1.0.0", "description": "Pets API"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/pets/{petId}": {
                # Path items may declare parameters shared by all their operations.
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "get": {
                    "operationId": "getPetById",
                    "description": "Get a pet",
                    "responses": {"200": {"description": "ok"}},
                },
            }
        },
    }
    llamaindex_tool_spec = OpenAPIToolSpec(spec=openapi_spec)
    deserialized = json.loads(llamaindex_tool_spec.load_openapi_spec()[0].text)
    endpoints = deserialized["endpoints"]

    assert [
        (endpoint["verb"], endpoint["path_template"]) for endpoint in endpoints
    ] == [("GET", "/pets/{petId}")]


def load_example_spec():
    current_file_path = Path(__file__).resolve()
    example_file = current_file_path.parent / "example.json"
    with example_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
