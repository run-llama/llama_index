import os
import pytest
from unittest import mock
from zhipuai.types.chat.chat_completion import (
    Completion,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
)
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.zhipuai import ZhipuAI


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
