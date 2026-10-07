from unittest.mock import MagicMock

import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.mistralai import MistralAI


def _mistralai(model: str) -> MistralAI:
    return MistralAI(api_key="fake-key", model=model)


def _mock_fim_response(llm: MistralAI, text: str) -> MagicMock:
    """Stub the FIM endpoint so no network call is made."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    client = MagicMock()
    client.fim.complete.return_value = response
    llm._client = client
    return client


def test_class():
    names_of_base_classes = [b.__name__ for b in MistralAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_fill_in_middle_accepts_dated_code_models():
    # Regression: "codestral-2508" was rejected because the guard was a
    # substring test against the single string "codestral-latest".
    llm = _mistralai("codestral-2508")
    client = _mock_fim_response(llm, "  return a + b")

    response = llm.fill_in_middle("def add(a, b):", "\n\nprint(add(1, 2))")

    assert response.text == "  return a + b"
    client.fim.complete.assert_called_once()


def test_fill_in_middle_accepts_the_latest_code_model():
    llm = _mistralai("codestral-latest")
    _mock_fim_response(llm, "pass")

    assert llm.fill_in_middle("def noop():", "").text == "pass"


def test_fill_in_middle_forwards_stop_sequences():
    llm = _mistralai("codestral-latest")
    client = _mock_fim_response(llm, "pass")

    llm.fill_in_middle("def noop():", "", stop=["\n\n"])

    assert client.fim.complete.call_args.kwargs["stop"] == ["\n\n"]


def test_fill_in_middle_rejects_non_code_models():
    llm = _mistralai("mistral-small-latest")
    client = _mock_fim_response(llm, "unused")

    with pytest.raises(ValueError, match="code model"):
        llm.fill_in_middle("def add(a, b):", "")

    client.fim.complete.assert_not_called()


def test_fill_in_middle_error_lists_the_supported_models():
    llm = _mistralai("mistral-small-latest")
    _mock_fim_response(llm, "unused")

    with pytest.raises(ValueError) as excinfo:
        llm.fill_in_middle("def add(a, b):", "")

    assert "codestral-2508" in str(excinfo.value)
