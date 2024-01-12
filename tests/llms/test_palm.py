"""Test PaLM."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest


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


from llama_index.core.llms.types import CompletionResponse
from llama_index.llms.palm import PaLM


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="PaLM requires Python 3.9 or higher"
)
def test_palm() -> None:
    """Test palm."""
    # Set up fake package here, as test_gemini uses the same package.
    sys.modules["google.generativeai"] = MockPalmPackage()

    palm = PaLM(api_key="test_api_key", model_name="palm_model")
    response = palm.complete("hello world")
    assert isinstance(response, CompletionResponse)
    assert response.text == "hello world"
