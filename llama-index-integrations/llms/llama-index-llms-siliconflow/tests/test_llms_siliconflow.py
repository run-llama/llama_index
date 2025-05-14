import json
import pytest
import types
from typing import Optional, Type
from unittest import mock
from requests import Response
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.siliconflow import SiliconFlow

RESPONSE_JSON = {
    "id": "<string>",
    "choices": [
        {
            "message": {"role": "assistant", "content": "<string>"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 123,
        "total_tokens": 123,
    },
    "created": 123,
    "model": "<string>",
    "object": "chat.completion",
}


class MockAsyncResponse:
    def __init__(self, json_data) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    async def __aenter__(self) -> "MockAsyncResponse":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[types.TracebackType],
    ) -> None:
        pass

    async def json(self) -> dict:
        return self._json_data


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in SiliconFlow.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_llm_model_alias():
    model = "deepseek-ai/DeepSeek-V2.5"
    api_key = "api_key_test"
    llm = SiliconFlow(model=model, api_key=api_key)
    assert llm.model == model
    assert llm.model_kwargs is not None


def test_llm_complete():
    input_text = "..."
    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = json.dumps(RESPONSE_JSON).encode("utf-8")
    expected_result = CompletionResponse(text="<string>", raw=RESPONSE_JSON)
    llm = SiliconFlow(api_key="...")
    with mock.patch("requests.Session.post", return_value=mock_response) as mock_post:
        actual_result = llm.complete(input_text)
        assert actual_result.text == expected_result.text
        assert actual_result.additional_kwargs == actual_result.additional_kwargs
        assert actual_result.raw == actual_result.raw
        assert actual_result.logprobs == actual_result.logprobs

        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": input_text}],
                "stream": False,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )


@pytest.mark.asyncio
async def test_llm_async_complete():
    input_text = "..."
    mock_response = MockAsyncResponse(json_data=RESPONSE_JSON)
    expected_result = CompletionResponse(text="<string>", raw=RESPONSE_JSON)
    llm = SiliconFlow(api_key="...")
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=mock_response
    ) as mock_post:
        actual_result = await llm.acomplete(input_text)

        assert actual_result.text == expected_result.text
        assert actual_result.additional_kwargs == actual_result.additional_kwargs
        assert actual_result.raw == actual_result.raw
        assert actual_result.logprobs == actual_result.logprobs

        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": input_text}],
                "stream": False,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )
