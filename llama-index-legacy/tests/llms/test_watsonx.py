import sys
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock

import pytest
from llama_index.legacy.core.llms.types import ChatMessage

try:
    import ibm_watson_machine_learning
except ImportError:
    ibm_watson_machine_learning = None

from llama_index.legacy.llms.watsonx import WatsonX


class MockStreamResponse:
    def __iter__(self) -> Generator[str, Any, None]:
        deltas = ["\n\nThis ", "is indeed", " a test"]
        yield from deltas


class MockIBMModelModule(MagicMock):
    class Model:
        def __init__(
            self,
            model_id: str,
            credentials: dict,
            project_id: Optional[str] = None,
            space_id: Optional[str] = None,
        ) -> None:
            pass

        def get_details(self) -> Dict[str, Any]:
            return {"model_details": "Mock IBM Watson Model"}

        def generate_text(self, prompt: str, params: Optional[dict] = None) -> str:
            return "\n\nThis is indeed a test"

        def generate_text_stream(
            self, prompt: str, params: Optional[dict] = None
        ) -> MockStreamResponse:
            return MockStreamResponse()


sys.modules[
    "ibm_watson_machine_learning.foundation_models.model"
] = MockIBMModelModule()


@pytest.mark.skipif(
    ibm_watson_machine_learning is None,
    reason="ibm-watson-machine-learning not installed",
)
def test_model_basic() -> None:
    credentials = {"url": "https://thisisa.fake.url/", "apikey": "fake_api_key"}
    project_id = "fake_project_id"

    test_prompt = "This is a test"
    llm = WatsonX(
        model_id="ibm/granite-13b-instruct-v1",
        credentials=credentials,
        project_id=project_id,
    )

    response = llm.complete(test_prompt)
    assert response.text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.skipif(
    ibm_watson_machine_learning is None,
    reason="ibm-watson-machine-learning not installed",
)
def test_model_streaming() -> None:
    credentials = {"url": "https://thisisa.fake.url/", "apikey": "fake_api_key"}
    project_id = "fake_project_id"

    test_prompt = "This is a test"
    llm = WatsonX(
        model_id="ibm/granite-13b-instruct-v1",
        credentials=credentials,
        project_id=project_id,
    )

    response_gen = llm.stream_complete(test_prompt)
    response = list(response_gen)

    assert response[-1].text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    chat_response = list(chat_response_gen)
    assert chat_response[-1].message.content == "\n\nThis is indeed a test"
