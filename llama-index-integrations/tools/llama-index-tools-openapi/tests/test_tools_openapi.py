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


def load_example_spec():
    current_file_path = Path(__file__).resolve()
    example_file = current_file_path.parent / "example.json"
    with example_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_circular_ref():
    circular_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test Circular API", "version": "1.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/nodes": {
                "get": {
                    "operationId": "getNodes",
                    "description": "Get hierarchical nodes",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Node"}
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "child": {"$ref": "#/components/schemas/Node"},
                    },
                }
            }
        },
    }
    llamaindex_tool_spec = OpenAPIToolSpec(spec=circular_spec)
    spec_array = llamaindex_tool_spec.load_openapi_spec()
    deserialized = json.loads(spec_array[0].text)
    assert len(deserialized["endpoints"]) == 1
    endpoint = deserialized["endpoints"][0]
    assert endpoint["verb"] == "GET"
    assert endpoint["path_template"] == "/nodes"


def test_indirect_circular_ref():
    circular_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test Indirect Circular API", "version": "1.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "getUsers",
                    "description": "Get users with circular org ownership",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "org": {"$ref": "#/components/schemas/Org"},
                    },
                },
                "Org": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "owner": {"$ref": "#/components/schemas/User"},
                    },
                },
            }
        },
    }
    llamaindex_tool_spec = OpenAPIToolSpec(spec=circular_spec)
    spec_array = llamaindex_tool_spec.load_openapi_spec()
    deserialized = json.loads(spec_array[0].text)
    assert len(deserialized["endpoints"]) == 1
    endpoint = deserialized["endpoints"][0]
    assert endpoint["verb"] == "GET"
    assert endpoint["path_template"] == "/users"
