"""Test PaLM."""

import sys
from unittest.mock import MagicMock
from typing import Any


def _mock_palm_completion(model_name: str, prompt: str, **kwargs: Any) -> str:
    """Mock PaLM completion."""
    completion = MagicMock()
    completion.result = prompt
    completion.candidates = [{"prompt": prompt}]
    return completion


class MockPalmPackage(MagicMock):
    """Mock PaLM package."""

    def _mock_models(self) -> Any:
        model = MagicMock()
        model.name = "palm_model"
        return [model]

    def generate_text(self, model: str, prompt: str, **kwargs: Any) -> str:
        """Mock PaLM completion."""
        return _mock_palm_completion(model, prompt, **kwargs)

    def list_models(self) -> Any:
        return self._mock_models()


sys.modules["google.generativeai"] = MockPalmPackage()


from llama_index.llms.palm import PaLM  # noqa: E402
from llama_index.llms.base import CompletionResponse  # noqa: E402
from typing import Any  # noqa: E402


def test_palm() -> None:
    """Test palm."""
    palm = PaLM(api_key="test_api_key", model_name="palm_model")
    response = palm.complete("hello world")
    assert isinstance(response, CompletionResponse)
    assert response.text == "hello world"
