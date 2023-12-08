import json
from io import BytesIO
from typing import Any, Generator

from botocore.response import StreamingBody
from botocore.stub import Stubber
from llama_index.llms import Bedrock
from llama_index.llms.types import ChatMessage
from pytest import MonkeyPatch


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


def get_invoke_model_response() -> dict:
    # response for titan model
    raw_stream_bytes = json.dumps(
        {
            "inputTextTokenCount": 3,
            "results": [
                {
                    "tokenCount": 14,
                    "outputText": "\n\nThis is indeed a test",
                    "completionReason": "FINISH",
                }
            ],
        }
    ).encode()
    raw_stream = BytesIO(raw_stream_bytes)
    content_length = len(raw_stream_bytes)

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
        "body": StreamingBody(
            raw_stream=raw_stream,
            content_length=content_length,
        ),
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


def test_model_basic() -> None:
    llm = Bedrock(
        model="amazon.titan-text-express-v1",
        profile_name=None,
        aws_region_name="us-east-1",
        aws_access_key_id="test",
    )

    bedrock_stubber = Stubber(llm._client)

    # response for llm.complete()
    bedrock_stubber.add_response(
        "invoke_model",
        get_invoke_model_response(),
    )
    # response for llm.chat()
    bedrock_stubber.add_response(
        "invoke_model",
        get_invoke_model_response(),
    )

    bedrock_stubber.activate()

    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert response.text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"

    bedrock_stubber.deactivate()


def test_model_streaming(monkeypatch: MonkeyPatch) -> None:
    # Cannot use Stubber to mock EventStream. See https://github.com/boto/botocore/issues/1621
    monkeypatch.setattr(
        "llama_index.llms.bedrock.completion_with_retry",
        mock_stream_completion_with_retry,
    )
    llm = Bedrock(
        model="amazon.titan-text-express-v1",
        profile_name=None,
        aws_region_name="us-east-1",
        aws_access_key_id="test",
    )
    test_prompt = "test prompt"
    response_gen = llm.stream_complete(test_prompt)
    response = list(response_gen)
    assert response[-1].text == "\n\nThis is indeed a test"

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    chat_response = list(chat_response_gen)
    assert chat_response[-1].message.content == "\n\nThis is indeed a test"
