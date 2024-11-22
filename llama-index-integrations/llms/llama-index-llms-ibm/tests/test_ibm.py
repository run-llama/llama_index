from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch
import warnings
import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.ibm import WatsonxLLM


def mock_return_guardrails_stats(*args) -> Dict:
    from ibm_watsonx_ai.foundation_models import ModelInference

    mock_client = MagicMock()
    mock_client._client.use_fm_ga_api = True
    return ModelInference._return_guardrails_stats(mock_client, args[0])


def mock_generate_with_hap(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        "model_id": "google/flan-ul2",
        "created_at": "2023-07-21T16:52:32.190Z",
        "results": [
            {
                "generated_text": "\n\nTEST",
                "generated_token_count": 1,
                "input_token_count": 12,
                "stop_reason": "eos_token",
                "moderations": {
                    "hap": [
                        {
                            "score": 0.8,
                            "input": False,
                            "position": {"start": 74, "end": 88},
                            "entity": "has_HAP",
                        }
                    ]
                },
            }
        ],
    }


def mock_generate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        "model_id": "google/flan-ul2",
        "created_at": "2023-07-21T16:52:32.190Z",
        "results": [
            {
                "generated_text": "\n\nTEST",
                "generated_token_count": 4,
                "input_token_count": 12,
                "stop_reason": "eos_token",
            }
        ],
    }


async def mock_agenerate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return mock_generate(args=args, kwargs=kwargs)


def mock_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        "model_id": "mistralai/mistral-large",
        "created_at": "2024-10-17T11:33:58.927Z",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nTEST",
                },
                "finish_reason": "stop",
            }
        ],
    }


async def mock_achat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return mock_chat(args=args, kwargs=kwargs)


def mock_stream_chat(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
    for content in ("I", " like", " it"):
        yield {
            "model_id": "mistralai/mistral-large",
            "created_at": "2024-10-17T11:41:26.140Z",
            "choices": [
                {"index": 0, "finish_reason": "stop", "delta": {"content": content}}
            ],
        }


def mock_completion_stream(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
    responses = [
        {
            "model_id": "google/flan-ul2",
            "created_at": "2024-05-13T17:32:03.326Z",
            "results": [
                {
                    "generated_text": "I",
                    "generated_token_count": 1,
                    "input_token_count": 0,
                    "stop_reason": "not_finished",
                }
            ],
        },
        {
            "model_id": "google/flan-ul2",
            "created_at": "2024-05-13T17:32:03.326Z",
            "results": [
                {
                    "generated_text": " like",
                    "generated_token_count": 2,
                    "input_token_count": 0,
                    "stop_reason": "not_finished",
                }
            ],
        },
        {
            "model_id": "google/flan-ul2",
            "created_at": "2024-05-13T17:32:03.348Z",
            "results": [
                {
                    "generated_text": " it",
                    "generated_token_count": 3,
                    "input_token_count": 0,
                    "stop_reason": "not_finished",
                }
            ],
        },
        {
            "model_id": "google/flan-ul2",
            "created_at": "2024-05-13T17:32:03.371Z",
            "results": [
                {
                    "generated_text": "",
                    "generated_token_count": 4,
                    "input_token_count": 0,
                    "stop_reason": "eos_token",
                }
            ],
        },
    ]
    yield from responses


def mock_completion_stream_text(
    *args: Any, **kwargs: Any
) -> Generator[str, None, None]:
    responses = ["I", " like", " it", ""]
    yield from responses


class TestWasonxLLMInference:
    TEST_URL = "https://us-south.ml.cloud.ibm.com"
    TEST_APIKEY = "12345"
    TEST_PROJECT_ID = "1234"

    TEST_MODEL = "google/flan-ul2"

    def test_initialization(self) -> None:
        with pytest.raises(ValueError, match=r"^Did not find") as e_info:
            _ = WatsonxLLM(model=self.TEST_MODEL, project_id=self.TEST_PROJECT_ID)

        # Cloud scenario
        with pytest.raises(
            ValueError, match=r"^Did not find 'apikey' or 'token',"
        ) as e_info:
            _ = WatsonxLLM(
                model_id=self.TEST_MODEL,
                url=self.TEST_URL,
                project_id=self.TEST_PROJECT_ID,
            )

        # CPD scenario
        with pytest.raises(ValueError, match=r"^Did not find instance_id") as e_info:
            _ = WatsonxLLM(
                model_id=self.TEST_MODEL,
                token="123",
                url="cpd-instance",
                project_id=self.TEST_PROJECT_ID,
            )

    @patch("llama_index.llms.ibm.base.ModelInference")
    def test_completion_model_basic(self, MockModelInference: MagicMock) -> None:
        mock_instance = MockModelInference.return_value

        mock_instance._return_guardrails_stats.side_effect = (
            mock_return_guardrails_stats
        )
        mock_instance.generate.return_value = mock_generate_with_hap()

        llm = WatsonxLLM(
            model=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )
        prompt = "test prompt"

        from ibm_watsonx_ai.foundation_models.utils.utils import HAPDetectionWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", category=HAPDetectionWarning)
            response = llm.complete(prompt, guardrails=True)

        assert len(w) == 1
        assert response.text == "\n\nTEST"

        mock_instance.chat.return_value = mock_chat()
        message = ChatMessage(role="user", content="test message")
        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nTEST"

    @patch("llama_index.llms.ibm.base.ModelInference")
    def test_completion_model_streaming_text(
        self, MockModelInference: MagicMock
    ) -> None:
        mock_instance = MockModelInference.return_value
        mock_instance.generate_text_stream.return_value = mock_completion_stream_text()

        llm = WatsonxLLM(
            model=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "I like it"

        mock_instance.chat_stream.return_value = mock_stream_chat()
        chat_response_stream = llm.stream_chat([message])
        chat_responses = list(chat_response_stream)
        assert chat_responses[-1].message.content == "I like it"

    @patch("llama_index.llms.ibm.base.ModelInference")
    def test_completion_model_streaming(self, MockModelInference: MagicMock) -> None:
        mock_instance = MockModelInference.return_value
        mock_instance.generate_text_stream.return_value = mock_completion_stream()
        mock_instance._return_guardrails_stats.side_effect = (
            mock_return_guardrails_stats
        )

        llm = WatsonxLLM(
            model=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt, raw_response=True)
        responses = list(response_gen)
        assert responses[-1].text == "I like it"

        mock_instance.chat_stream.return_value = mock_stream_chat()
        chat_response_stream = llm.stream_chat([message], raw_response=True)
        chat_responses = list(chat_response_stream)
        assert chat_responses[-1].message.content == "I like it"

    @pytest.mark.asyncio()
    @patch("llama_index.llms.ibm.base.ModelInference")
    async def test_complete_async(self, MockModelInference: MagicMock) -> None:
        mock_instance = MockModelInference.return_value
        mock_instance._return_guardrails_stats.side_effect = (
            mock_return_guardrails_stats
        )
        mock_instance.agenerate.return_value = mock_agenerate()
        watsonxllm = WatsonxLLM(
            model=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )

        response = await watsonxllm.acomplete("What do you think about Gen AI?")
        assert response.text == "\n\nTEST"

        mock_instance.achat.return_value = mock_achat()
        message = ChatMessage(role="user", content="test message")
        chat_response = await watsonxllm.achat([message])
        assert chat_response.message.content == "\n\nTEST"

    @pytest.mark.asyncio()
    @patch("llama_index.llms.ibm.base.ModelInference")
    async def test_stream_async(self, MockModelInference: MagicMock) -> None:
        mock_instance = MockModelInference.return_value
        mock_instance.generate_text_stream.return_value = mock_completion_stream()
        mock_instance._return_guardrails_stats.side_effect = (
            mock_return_guardrails_stats
        )

        llm = WatsonxLLM(
            model=self.TEST_MODEL,
            url=self.TEST_URL,
            apikey=self.TEST_APIKEY,
            project_id=self.TEST_PROJECT_ID,
        )
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = await llm.astream_complete(prompt, raw_response=True)
        responses = [el async for el in response_gen]
        assert responses[-1].text == "I like it"

        mock_instance.chat_stream.return_value = mock_stream_chat()
        chat_response_stream = await llm.astream_chat([message], raw_response=True)
        chat_responses = [el async for el in chat_response_stream]
        assert chat_responses[-1].message.content == "I like it"
