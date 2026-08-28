import os
import pytest
from unittest import mock
from zhipuai.types.chat.chat_completion import (
    Completion,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
)
from zhipuai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.zhipuai import ZhipuAI


def _stream_chunk(role, content, index=0):
    """
    Build a streaming chunk mirroring the OpenAI-compatible ZhipuAI stream.

    The first chunk carries ``delta.role="assistant"`` while subsequent
    content chunks have ``delta.role=None`` (the real-world behavior).
    """
    return ChatCompletionChunk(
        id="chatcmpl-test",
        created=1703487403,
        model="glm-4",
        choices=[
            Choice(
                index=index,
                delta=ChoiceDelta(role=role, content=content, tool_calls=None),
                finish_reason=None,
            )
        ],
        extra_json={},
    )


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in ZhipuAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_zhipuai_llm_model_alias():
    model = "glm-test"
    api_key = "api_key_test"
    llm = ZhipuAI(model=model, api_key=api_key)
    assert llm.model == model
    assert llm.model_kwargs is not None


def test_zhipuai_llm_metadata():
    api_key = "api_key_test"
    llm = ZhipuAI(model="glm-4", api_key=api_key)
    assert llm.metadata.is_function_calling_model is True
    llm = ZhipuAI(model="glm-4v", api_key=api_key)
    assert llm.metadata.is_function_calling_model is False


def test_zhipuai_completions_with_stop():
    mock_response = Completion(
        model="glm-4",
        created=1703487403,
        choices=[
            CompletionChoice(
                index=0,
                finish_reason="stop",
                message=CompletionMessage(
                    role="assistant",
                    content="MOCK_RESPONSE",
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=31, completion_tokens=217, total_tokens=248
        ),
    )
    predict_response = CompletionResponse(
        text="MOCK_RESPONSE",
        additional_kwargs={"tool_calls": []},
        raw=mock_response,
    )
    llm = ZhipuAI(model="glm-4", api_key="__fake_key__")
    with mock.patch.object(
        llm._client.chat.completions, "create", return_value=mock_response
    ):
        actual_chat = llm.complete("__query__", stop=["stop_words"])
        assert actual_chat == predict_response


def test_zhipuai_stream_chat_role_none_on_later_chunks():
    """
    Later stream chunks have delta.role=None and must not raise.

    Regression test: ``ChatMessage.role`` is a non-Optional ``MessageRole``,
    so passing ``role=None`` (as later chunks deliver) raised a pydantic
    ValidationError before the ``or MessageRole.ASSISTANT`` fallback was added.
    """
    chunks = [
        _stream_chunk(role="assistant", content="Hello"),
        _stream_chunk(role=None, content=" there"),
        _stream_chunk(role=None, content="!"),
    ]

    llm = ZhipuAI(model="glm-4", api_key="__fake_key__")
    with mock.patch.object(
        llm._client.chat.completions, "create", return_value=iter(chunks)
    ):
        responses = list(
            llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")])
        )

    assert len(responses) == len(chunks)
    for response in responses:
        assert response.message.role == MessageRole.ASSISTANT
    assert responses[-1].message.content == "Hello there!"


@pytest.mark.asyncio
async def test_zhipuai_astream_chat_role_none_on_later_chunks():
    """Async streaming must also tolerate delta.role=None on later chunks."""
    chunks = [
        _stream_chunk(role="assistant", content="Hello"),
        _stream_chunk(role=None, content=" there"),
        _stream_chunk(role=None, content="!"),
    ]

    llm = ZhipuAI(model="glm-4", api_key="__fake_key__")
    with mock.patch.object(
        llm._client.chat.completions, "create", return_value=iter(chunks)
    ):
        responses = [
            response
            async for response in await llm.astream_chat(
                [ChatMessage(role=MessageRole.USER, content="hi")]
            )
        ]

    assert len(responses) == len(chunks)
    for response in responses:
        assert response.message.role == MessageRole.ASSISTANT
    assert responses[-1].message.content == "Hello there!"


@pytest.mark.skipif(
    os.getenv("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
def test_completion():
    model = "glm-4"
    api_key = os.getenv("ZHIPUAI_API_KEY")
    llm = ZhipuAI(model=model, api_key=api_key)
    assert llm.complete("who are you")


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
async def test_async_completion():
    model = "glm-4"
    api_key = os.getenv("ZHIPUAI_API_KEY")
    llm = ZhipuAI(model=model, api_key=api_key)
    assert await llm.acomplete("who are you")
