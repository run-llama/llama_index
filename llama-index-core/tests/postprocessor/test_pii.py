"""Tests for the PII postprocessor's handling of malformed LLM responses."""

import json

import pytest

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import PIINodePostprocessor


class NoMarkerMockLLM(MockLLM):
    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text="I cannot process this request.")


class BadJSONMockLLM(MockLLM):
    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(
            text='Masked text\nOutput Mapping:\n{"NAME1": "Zhang Wei"'
        )


class GoodMockLLM(MockLLM):
    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(
            text='Masked text\nOutput Mapping:\n{"NAME1": "Zhang Wei"}'
        )


def test_mask_pii_happy_path():
    postprocessor = PIINodePostprocessor(llm=GoodMockLLM())
    text, mapping = postprocessor.mask_pii("Hello Zhang Wei")

    assert text == "Masked text"
    assert mapping == {"NAME1": "Zhang Wei"}


def test_mask_pii_missing_marker_raises_value_error():
    postprocessor = PIINodePostprocessor(llm=NoMarkerMockLLM())

    with pytest.raises(ValueError, match="Output Mapping:"):
        postprocessor.mask_pii("Hello Zhang Wei")


def test_mask_pii_invalid_json_raises_value_error():
    postprocessor = PIINodePostprocessor(llm=BadJSONMockLLM())

    with pytest.raises(ValueError, match="invalid JSON"):
        postprocessor.mask_pii("Hello Zhang Wei")


def test_mask_pii_error_includes_raw_response():
    postprocessor = PIINodePostprocessor(llm=NoMarkerMockLLM())
    raw = "I cannot process this request."

    with pytest.raises(ValueError) as excinfo:
        postprocessor.mask_pii("Hello Zhang Wei")

    assert raw in str(excinfo.value)


def test_mask_pii_json_roundtrip():
    postprocessor = PIINodePostprocessor(llm=GoodMockLLM())
    _, mapping = postprocessor.mask_pii("Hello Zhang Wei")

    assert json.dumps(mapping) == '{"NAME1": "Zhang Wei"}'
