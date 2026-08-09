"""PII postprocessor tests."""

import pytest
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import PIINodePostprocessor


class NoMarkerMockLLM(MockLLM):
    """Mock LLM that returns a response with no 'Output Mapping:' marker."""

    def complete(self, prompt, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(text="I cannot process this request.")


class InvalidJsonMockLLM(MockLLM):
    """Mock LLM that returns a marker followed by invalid JSON."""

    def complete(self, prompt, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(
            text="Hello [NAME1].\nOutput Mapping:\nnot valid json at all"
        )


class ValidMockLLM(MockLLM):
    """Mock LLM that returns a well-formed response."""

    def complete(self, prompt, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(
            text='Hello [NAME1].\nOutput Mapping:\n{"NAME1": "Zhang Wei"}'
        )


def test_mask_pii_missing_marker_raises_value_error() -> None:
    """Response without an 'Output Mapping:' marker should raise a clear ValueError."""
    postprocessor = PIINodePostprocessor(llm=NoMarkerMockLLM())
    with pytest.raises(ValueError, match="Output Mapping"):
        postprocessor.mask_pii("Some context with PII")


def test_mask_pii_invalid_json_raises_value_error() -> None:
    """Response with malformed JSON after the marker should raise a clear ValueError."""
    postprocessor = PIINodePostprocessor(llm=InvalidJsonMockLLM())
    with pytest.raises(ValueError, match="not valid JSON"):
        postprocessor.mask_pii("Some context with PII")


def test_mask_pii_valid_response() -> None:
    """A well-formed response should still parse correctly."""
    postprocessor = PIINodePostprocessor(llm=ValidMockLLM())
    text_output, json_dict = postprocessor.mask_pii("Some context with PII")
    assert text_output == "Hello [NAME1]."
    assert json_dict == {"NAME1": "Zhang Wei"}
