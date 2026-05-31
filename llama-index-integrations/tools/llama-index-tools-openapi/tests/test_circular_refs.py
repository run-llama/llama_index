"""
Tests for circular ``$ref`` handling in :class:`OpenAPIToolSpec`.

Regression coverage for run-llama/llama_index#15011: OpenAPI specs that
contain self-referential or mutually-referential ``$ref`` chains used to send
the recursive dereferencer into an infinite loop (raising ``RecursionError``).
"""

import json

import pytest

from llama_index.tools.openapi import OpenAPIToolSpec


def _minimal_paths() -> dict:
    """Return a tiny ``paths`` block that the spec processor accepts."""
    return {
        "/things": {
            "get": {
                "operationId": "listThings",
                "summary": "List things",
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Thing"}
                            }
                        },
                    }
                },
            }
        }
    }


def _base_spec(schemas: dict) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "t", "version": "1", "description": "d"},
        "servers": [{"url": "http://example.com"}],
        "paths": _minimal_paths(),
        "components": {"schemas": schemas},
    }


def test_self_referential_schema_does_not_recurse_forever():
    """A schema referencing itself must not blow the stack."""
    spec = _base_spec(
        {
            "Thing": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    # Self-reference -- a Thing contains a child Thing.
                    "child": {"$ref": "#/components/schemas/Thing"},
                },
            }
        }
    )

    tool = OpenAPIToolSpec(spec=spec)
    payload = json.loads(tool.spec.text)

    # The endpoint is preserved.
    assert len(payload["endpoints"]) == 1
    endpoint = payload["endpoints"][0]
    assert endpoint["path_template"] == "/things"

    # The response schema is materialised at least one level deep, and the
    # recursive child position is collapsed rather than expanded forever.
    schema = endpoint["responses"]["content"]["application/json"]["schema"]
    assert schema["type"] == "object"
    assert "child" in schema["properties"]


def test_mutually_recursive_schemas_terminate():
    """Two schemas pointing at each other must terminate dereferencing."""
    spec = _base_spec(
        {
            "A": {
                "type": "object",
                "properties": {"b": {"$ref": "#/components/schemas/B"}},
            },
            "B": {
                "type": "object",
                "properties": {"a": {"$ref": "#/components/schemas/A"}},
            },
        }
    )
    # Make the endpoint return an ``A`` so the cycle is reachable.
    spec["paths"]["/things"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ] = {"$ref": "#/components/schemas/A"}

    tool = OpenAPIToolSpec(spec=spec)
    payload = json.loads(tool.spec.text)
    schema = payload["endpoints"][0]["responses"]["content"]["application/json"][
        "schema"
    ]
    assert schema["type"] == "object"
    # The cycle was broken: ``b`` exists but the inner ``a`` does not expand
    # back into the full ``A`` definition.
    assert "b" in schema["properties"]


def test_acyclic_refs_still_fully_dereferenced():
    """Non-circular ``$ref`` chains must continue to resolve as before."""
    spec = _base_spec(
        {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "Thing": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"$ref": "#/components/schemas/Address"},
                },
            },
        }
    )
    tool = OpenAPIToolSpec(spec=spec)
    payload = json.loads(tool.spec.text)
    schema = payload["endpoints"][0]["responses"]["content"]["application/json"][
        "schema"
    ]
    # ``$ref`` should be resolved -- no leftover ``$ref`` key.
    assert "$ref" not in schema
    assert schema["properties"]["address"]["type"] == "object"
    assert schema["properties"]["address"]["properties"]["city"]["type"] == "string"


@pytest.mark.parametrize("depth", [200, 1500])
def test_deep_self_reference_does_not_raise_recursion_error(depth):
    """Even with the recursion limit lowered, the dereferencer terminates."""
    import sys

    spec = _base_spec(
        {
            "Thing": {
                "type": "object",
                "properties": {"child": {"$ref": "#/components/schemas/Thing"}},
            }
        }
    )

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(depth, 200))
    try:
        # Must not raise RecursionError.
        OpenAPIToolSpec(spec=spec)
    finally:
        sys.setrecursionlimit(original_limit)
