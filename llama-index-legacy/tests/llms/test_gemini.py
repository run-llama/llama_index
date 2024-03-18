import sys
import types
from typing import Any, Mapping
from unittest import mock

import pytest
from llama_index.legacy.llms.base import CompletionResponse
from llama_index.legacy.llms.gemini import Gemini


class FakeGoogleDataclass(types.SimpleNamespace):
    """Emulate the dataclasses used in the genai package."""

    def __init__(self, d: Mapping[str, Any], *args: Any, **kwargs: Any):
        self.d = d
        super().__init__(**d)

    def to_dict(self) -> Mapping[str, Any]:
        return self.d


class MockGenaiPackage(mock.Mock):
    """Stubbed-out google.generativeai package."""

    response_text = "default response"

    def get_model(self, name: str, **kwargs: Any) -> Any:
        model = mock.Mock()
        model.name = name
        model.supported_generation_methods = ["generateContent"]
        model.input_token_limit = 4321
        model.output_token_limit = 12345
        return model

    def _gen_content(
        self, contents: Any, *, stream: bool = False, **kwargs: Any
    ) -> Any:
        content = mock.Mock()
        content.text = self.response_text
        content.candidates = [
            FakeGoogleDataclass(
                {
                    "content": {
                        "parts": [{"text": self.response_text}],
                        "role": "model",
                    },
                    "finish_reason": 1,
                }
            )
        ]
        content.prompt_feedback = FakeGoogleDataclass({})

        if stream:
            # Can't yield-from here as this function is called as a mock side effect.
            return [content]
        else:
            return content

    def GenerativeModel(self, **kwargs: Any) -> Any:
        gmodel = mock.Mock()
        gmodel.generate_content.side_effect = self._gen_content
        return gmodel


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Gemini supports Python 3.9+")
def test_gemini() -> None:
    # Set up fake package here, as test_palm uses the same package.
    sys.modules["google.generativeai"] = MockGenaiPackage()

    MockGenaiPackage.response_text = "echo echo"

    llm = Gemini(model_name="models/one")
    response = llm.complete("say echo")

    assert isinstance(response, CompletionResponse)
    assert response.text == "echo echo"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Gemini supports Python 3.9+")
def test_gemini_stream() -> None:
    # Set up fake package here, as test_palm uses the same package.
    sys.modules["google.generativeai"] = MockGenaiPackage()

    MockGenaiPackage.response_text = "echo echo"

    llm = Gemini(model_name="models/one")
    (response,) = llm.stream_complete("say echo")

    assert isinstance(response, CompletionResponse)
    assert response.text == "echo echo"
