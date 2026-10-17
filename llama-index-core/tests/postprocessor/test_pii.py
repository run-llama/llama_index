from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import PIINodePostprocessor


def _llm_with(text: str) -> MockLLM:
    class _FixedLLM(MockLLM):
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            return CompletionResponse(text=text)

    return _FixedLLM()


def test_mask_pii_missing_mapping_does_not_indexerror() -> None:
    processor = PIINodePostprocessor(
        llm=_llm_with("Hello [NAME1], I am [NAME2].")
    )
    text, mapping = processor.mask_pii("Hello Ada, I am Bob.")
    assert "Hello [NAME1], I am [NAME2]." in text
    assert mapping == {}


def test_mask_pii_parses_mapping_json() -> None:
    processor = PIINodePostprocessor(
        llm=_llm_with('Hello [NAME1].\nOutput Mapping:\n{"NAME1": "Ada"}')
    )
    text, mapping = processor.mask_pii("Hello Ada.")
    assert mapping == {"NAME1": "Ada"}
    assert "Hello [NAME1]." in text


def test_mask_pii_invalid_mapping_json_returns_empty_dict() -> None:
    processor = PIINodePostprocessor(
        llm=_llm_with("Hello [NAME1].\nOutput Mapping:\nnot-json")
    )
    _, mapping = processor.mask_pii("Hello Ada.")
    assert mapping == {}
