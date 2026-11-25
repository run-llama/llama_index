"""Tests for response schema dataclasses."""

import copy
import pickle
from typing import Optional

import pytest
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import NodeWithScore, TextNode


class SampleModel(BaseModel):
    """Sample Pydantic model for PydanticResponse tests."""

    name: str
    age: Optional[int] = None


def test_pydantic_response_copy() -> None:
    """Shallow-copying PydanticResponse should not trigger RecursionError."""
    resp = PydanticResponse(
        response=SampleModel(name="Alice", age=30),
        source_nodes=[NodeWithScore(node=TextNode(text="ctx"), score=0.9)],
        metadata={"source": "unit-test"},
    )
    copied = copy.copy(resp)
    assert copied == resp
    assert copied is not resp
    assert copied.name == "Alice"
    assert copied.age == 30

    empty_resp = PydanticResponse(response=None)
    copied_empty = copy.copy(empty_resp)
    assert copied_empty == empty_resp
    assert copied_empty is not empty_resp


def test_pydantic_response_deepcopy() -> None:
    """Deep-copying PydanticResponse should not trigger RecursionError."""
    resp = PydanticResponse(
        response=SampleModel(name="Alice", age=30),
        source_nodes=[NodeWithScore(node=TextNode(text="ctx"), score=0.9)],
        metadata={"nested": {"key": "value"}},
    )
    copied = copy.deepcopy(resp)
    assert copied == resp
    assert copied is not resp
    assert copied.response is not resp.response
    assert copied.metadata is not resp.metadata
    assert copied.name == "Alice"

    empty_resp = PydanticResponse(response=None)
    copied_empty = copy.deepcopy(empty_resp)
    assert copied_empty == empty_resp
    assert copied_empty is not empty_resp


def test_pydantic_response_pickle_roundtrip() -> None:
    """Pickling and unpickling PydanticResponse should succeed without RecursionError."""
    resp = PydanticResponse(
        response=SampleModel(name="Bob", age=42),
        source_nodes=[NodeWithScore(node=TextNode(text="doc"), score=0.8)],
        metadata={"k": "v"},
    )
    restored = pickle.loads(pickle.dumps(resp))
    assert isinstance(restored, PydanticResponse)
    assert restored == resp
    assert restored.name == "Bob"
    assert restored.age == 42


def test_pydantic_response_uninitialized_instance_no_recursion() -> None:
    """Uninitialized instances created via __new__ must not recurse in __getattr__."""
    uninit = PydanticResponse.__new__(PydanticResponse)
    assert uninit.response is None
    assert uninit.missing_field is None
    with pytest.raises(AttributeError):
        _ = uninit.__setstate__


def test_pydantic_response_getattr_forwarding_and_dunder_errors() -> None:
    """Field forwarding and None fallback are preserved while dunders raise AttributeError."""
    resp = PydanticResponse(response=SampleModel(name="Carol", age=25))
    assert resp.name == "Carol"
    assert resp.age == 25
    assert resp.nonexistent is None
    assert PydanticResponse(response=None).name is None

    for dunder in ("__copy__", "__deepcopy__", "__setstate__", "__unknown_dunder__"):
        assert not hasattr(resp, dunder)
        with pytest.raises(AttributeError):
            getattr(resp, dunder)
