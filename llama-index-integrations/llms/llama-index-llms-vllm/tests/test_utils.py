import json
from types import SimpleNamespace

from llama_index.llms.vllm.utils import get_response


def _build_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(content=json.dumps(payload).encode("utf-8"))


def test_get_response_handles_legacy_text_field() -> None:
    resp = _build_response({"text": ["hello world"]})
    assert get_response(resp) == ["hello world"]


def test_get_response_handles_choices_format() -> None:
    resp = _build_response(
        {
            "choices": [
                {"text": "choice-1"},
                {"text": "choice-2"},
            ]
        }
    )
    assert get_response(resp) == ["choice-1", "choice-2"]
