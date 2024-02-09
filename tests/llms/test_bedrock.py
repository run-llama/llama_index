import json
from io import BytesIO
from typing import Any, Generator

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from llama_index.core.llms.types import ChatMessage
from llama_index.llms import Bedrock
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


def get_invoke_model_response(payload: str) -> dict:
    raw_stream_bytes = payload.encode()
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


class MockStreamCompletionWithRetry:
    def __init__(self, expected_prompt: str):
        self.expected_prompt = expected_prompt

    def mock_stream_completion_with_retry(
        self, request_body: str, *args: Any, **kwargs: Any
    ) -> dict:
        assert json.loads(request_body) == {
            "inputText": self.expected_prompt,
            "textGenerationConfig": {"maxTokenCount": 512, "temperature": 0.1},
        }
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


@pytest.mark.parametrize(
    ("model", "complete_request", "response_body", "chat_request"),
    [
        (
            "amazon.titan-text-express-v1",
            '{"inputText": "test prompt", "textGenerationConfig": {"temperature": 0.1, "maxTokenCount": 512}}',
            '{"inputTextTokenCount": 3, "results": [{"tokenCount": 14, "outputText": "\\n\\nThis is indeed a test", "completionReason": "FINISH"}]}',
            '{"inputText": "user: test prompt\\nassistant: ", "textGenerationConfig": {"temperature": 0.1, "maxTokenCount": 512}}',
        ),
        (
            "ai21.j2-grande-instruct",
            '{"prompt": "test prompt", "temperature": 0.1, "maxTokens": 512}',
            '{"completions": [{"data": {"text": "\\n\\nThis is indeed a test"}}]}',
            '{"prompt": "user: test prompt\\nassistant: ", "temperature": 0.1, "maxTokens": 512}',
        ),
        (
            "cohere.command-text-v14",
            '{"prompt": "test prompt", "temperature": 0.1, "max_tokens": 512}',
            '{"generations": [{"text": "\\n\\nThis is indeed a test"}]}',
            '{"prompt": "user: test prompt\\nassistant: ", "temperature": 0.1, "max_tokens": 512}',
        ),
        (
            "anthropic.claude-instant-v1",
            '{"prompt": "\\n\\nHuman: test prompt\\n\\nAssistant: ", "temperature": 0.1, "max_tokens_to_sample": 512}',
            '{"completion": "\\n\\nThis is indeed a test"}',
            '{"prompt": "\\n\\nHuman: test prompt\\n\\nAssistant: ", "temperature": 0.1, "max_tokens_to_sample": 512}',
        ),
        (
            "meta.llama2-13b-chat-v1",
            '{"prompt": "<s> [INST] <<SYS>>\\n You are a helpful, respectful and '
            "honest assistant. Always answer as helpfully as possible and follow "
            "ALL given instructions. Do not speculate or make up information. Do "
            "not reference any given instructions or context. \\n<</SYS>>\\n\\n "
            'test prompt [/INST]", "temperature": 0.1, "max_gen_len": 512}',
            '{"generation": "\\n\\nThis is indeed a test"}',
            '{"prompt": "<s> [INST] <<SYS>>\\n You are a helpful, respectful and '
            "honest assistant. Always answer as helpfully as possible and follow "
            "ALL given instructions. Do not speculate or make up information. Do "
            "not reference any given instructions or context. \\n<</SYS>>\\n\\n "
            'test prompt [/INST]", "temperature": 0.1, "max_gen_len": 512}',
        ),
    ],
)
def test_model_basic(
    model: str, complete_request: str, response_body: str, chat_request: str
) -> None:
    llm = Bedrock(
        model=model,
        profile_name=None,
        region_name="us-east-1",
        aws_access_key_id="test",
    )

    bedrock_stubber = Stubber(llm._client)

    # response for llm.complete()
    bedrock_stubber.add_response(
        "invoke_model",
        get_invoke_model_response(response_body),
        {"body": complete_request, "modelId": model},
    )
    # response for llm.chat()
    bedrock_stubber.add_response(
        "invoke_model",
        get_invoke_model_response(response_body),
        {"body": chat_request, "modelId": model},
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
    monkeypatch.setattr(
        "llama_index.llms.bedrock.completion_with_retry",
        MockStreamCompletionWithRetry("test prompt").mock_stream_completion_with_retry,
    )
    llm = Bedrock(
        model="amazon.titan-text-express-v1",
        profile_name=None,
        region_name="us-east-1",
        aws_access_key_id="test",
    )
    test_prompt = "test prompt"
    response_gen = llm.stream_complete(test_prompt)
    response = list(response_gen)
    assert response[-1].text == "\n\nThis is indeed a test"

    monkeypatch.setattr(
        "llama_index.llms.bedrock.completion_with_retry",
        MockStreamCompletionWithRetry(
            "user: test prompt\nassistant: "
        ).mock_stream_completion_with_retry,
    )
    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    chat_response = list(chat_response_gen)
    assert chat_response[-1].message.content == "\n\nThis is indeed a test"
