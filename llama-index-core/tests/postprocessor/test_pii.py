"""Test PII postprocessor."""

import pytest

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import PIINodePostprocessor


class _FixedResponseMockLLM(MockLLM):
    """MockLLM that always returns a fixed completion, ignoring the prompt."""

    def __init__(self, fixed_response: str) -> None:
        super().__init__()
        self._fixed_response = fixed_response

    def complete(self, prompt, formatted=False, **kwargs):
        return CompletionResponse(text=self._fixed_response)


def test_mask_pii_raises_clear_error_when_output_mapping_marker_missing():
    """
    An LLM response that doesn't follow the expected 'Output Mapping:' format
    (e.g. a refusal, or a model that ignores instructions) must fail with a clear
    ValueError, not a raw IndexError from unpacking a 1-element split().
    """
    llm = _FixedResponseMockLLM(fixed_response="I cannot process this request.")
    postprocessor = PIINodePostprocessor(llm=llm)

    with pytest.raises(ValueError, match="Output Mapping"):
        postprocessor.mask_pii("Some context containing PII")


def test_mask_pii_raises_clear_error_on_invalid_json_mapping():
    """
    A response with the marker present but a malformed/non-JSON mapping section
    must fail with a clear ValueError, not a raw JSONDecodeError.
    """
    llm = _FixedResponseMockLLM(
        fixed_response="Hello [NAME1].\nOutput Mapping:\nnot valid json"
    )
    postprocessor = PIINodePostprocessor(llm=llm)

    with pytest.raises(ValueError, match="JSON"):
        postprocessor.mask_pii("Some context containing PII")


def test_mask_pii_succeeds_with_well_formed_response():
    llm = _FixedResponseMockLLM(
        fixed_response=('Hello [NAME1].\nOutput Mapping:\n{"NAME1": "Zhang Wei"}')
    )
    postprocessor = PIINodePostprocessor(llm=llm)

    text_output, mapping = postprocessor.mask_pii("Hello Zhang Wei.")
    assert text_output == "Hello [NAME1]."
    assert mapping == {"NAME1": "Zhang Wei"}
