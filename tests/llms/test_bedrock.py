from typing import Any, Generator

import pytest
from llama_index.llms.base import ChatMessage
from pytest import MonkeyPatch

try:
    import boto3
except ImportError:
    boto3 = None
from llama_index.llms import Bedrock


class MockStreamingBody:
    def read(self) -> str:
        return """{
        "inputTextTokenCount": 3,
        "results": [
        {"tokenCount": 14,
        "outputText": "\\n\\nThis is indeed a test",
        "completionReason": "FINISH"
        }]}
        """


class MockEventStream:
    def __iter__(self) -> Generator[dict, None, None]:
        deltas = [b"\\n\\nThis ", b"is indeed", b" a test"]
        for delta in deltas:
            yield {
                "chunk": {
                    "bytes": b'{"outputText":"' + delta + b'",'
                    b'"index":0,"totalOutputTextTokenCount":20,'
                    b'"completionReason":"LENGTH","inputTextTokenCount":7}'
                }
            }


def mock_completion_with_retry(*args: Any, **kwargs: Any) -> dict:
    return {
        "ResponseMetadata": {
            "HTTPHeaders": {
                "connection": "keep-alive",
                "content-length": "246",
                "content-type": "application/json",
                "date": "Fri, 20 Oct 2023 08:20:44 GMT",
                "x-amzn-requestid": "667dq648-fbc3-4a7b-8f0e-4575f1f1f11d",
            },
            "HTTPStatusCode": 200,
            "RequestId": "667dq648-fbc3-4a7b-8f0e-4575f1f1f11d",
            "RetryAttempts": 0,
        },
        "body": MockStreamingBody(),
        "contentType": "application/json",
    }


def mock_stream_completion_with_retry(*args: Any, **kwargs: Any) -> dict:
    return {
        "ResponseMetadata": {
            "HTTPHeaders": {
                "connection": "keep-alive",
                "content-type": "application/vnd.amazon.eventstream",
                "date": "Fri, 20 Oct 2023 11:59:03 GMT",
                "transfer-encoding": "chunked",
                "x-amzn-bedrock-content-type": "application/json",
                "x-amzn-requestid": "ef9af51b-7ba5-4020-3793-f4733226qb84",
            },
            "HTTPStatusCode": 200,
            "RequestId": "ef9af51b-7ba5-4020-3793-f4733226qb84",
            "RetryAttempts": 0,
        },
        "body": MockEventStream(),
        "contentType": "application/json",
    }


@pytest.mark.skipif(boto3 is None, reason="bedrock not installed")
def test_model_basic(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.bedrock.completion_with_retry", mock_completion_with_retry
    )
    llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=None)
    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert response.text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.skipif(boto3 is None, reason="bedrock not installed")
def test_model_streaming(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.bedrock.completion_with_retry",
        mock_stream_completion_with_retry,
    )
    llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=None)
    test_prompt = "test prompt"
    response_gen = llm.stream_complete(test_prompt)
    response = list(response_gen)
    assert response[-1].text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    chat_response = list(chat_response_gen)
    assert chat_response[-1].message.content == "\n\nThis is indeed a test"
