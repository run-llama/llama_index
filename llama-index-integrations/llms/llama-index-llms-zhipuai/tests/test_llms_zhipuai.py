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


def test_glm_5_3_models_metadata():
    api_key = "api_key_test"
    # GLM-5.3 flagship
    llm = ZhipuAI(model="glm-5.3", api_key=api_key)
    assert llm.metadata.context_window == 128_000
    assert llm.metadata.is_function_calling_model is True

    # GLM-5.3-Flash
    llm_flash = ZhipuAI(model="glm-5.3-flash", api_key=api_key)
    assert llm_flash.metadata.context_window == 128_000
    assert llm_flash.metadata.is_function_calling_model is True

    # GLM-5V-Turbo
    llm_v = ZhipuAI(model="glm-5v-turbo", api_key=api_key)
    assert llm_v.metadata.context_window == 200_000
    assert llm_v.metadata.is_function_calling_model is True

    # GLM-4V-Plus
    llm_4vp = ZhipuAI(model="glm-4v-plus", api_key=api_key)
    assert llm_4vp.metadata.context_window == 128_000
    assert llm_4vp.metadata.is_function_calling_model is True


def test_glm_case_and_prefix_resilience():
    api_key = "api_key_test"
    # Upper-case aliases
    llm_upper = ZhipuAI(model="GLM-5.3", api_key=api_key)
    assert llm_upper.metadata.context_window == 128_000

    llm_flash_upper = ZhipuAI(model="GLM-5.3-Flash", api_key=api_key)
    assert llm_flash_upper.metadata.context_window == 128_000

    llm_v_upper = ZhipuAI(model="GLM-5V-Turbo", api_key=api_key)
    assert llm_v_upper.metadata.context_window == 200_000

    # Prefix-qualified models (e.g., zai/glm-5.3)
    llm_prefix = ZhipuAI(model="zai/glm-5.3", api_key=api_key)
    assert llm_prefix.metadata.context_window == 128_000


def test_unknown_model_safe_fallback():
    api_key = "api_key_test"
    # Should not raise ValueError, falls back gracefully to DEFAULT_CONTEXT_WINDOW
    llm_custom = ZhipuAI(model="custom-finetuned-glm", api_key=api_key)
    assert llm_custom.metadata.context_window == 128_000


def test_api_key_environment_fallbacks():
    # 1. ZHIPUAI_API_KEY
    with mock.patch.dict(
        os.environ, {"ZHIPUAI_API_KEY": "key_from_zhipuai_env"}, clear=True
    ):
        llm = ZhipuAI(model="glm-5.3")
        assert llm.api_key == "key_from_zhipuai_env"

    # 2. BIGMODEL_API_KEY
    with mock.patch.dict(
        os.environ, {"BIGMODEL_API_KEY": "key_from_bigmodel_env"}, clear=True
    ):
        llm = ZhipuAI(model="glm-5.3")
        assert llm.api_key == "key_from_bigmodel_env"

    # 3. ZAI_API_KEY
    with mock.patch.dict(os.environ, {"ZAI_API_KEY": "key_from_zai_env"}, clear=True):
        llm = ZhipuAI(model="glm-5.3")
        assert llm.api_key == "key_from_zai_env"

    # 4. Explicit parameter overrides environment variable
    with mock.patch.dict(os.environ, {"ZHIPUAI_API_KEY": "env_key"}, clear=True):
        llm = ZhipuAI(model="glm-5.3", api_key="explicit_key")
        assert llm.api_key == "explicit_key"


def test_base_url_configuration():
    # Explicit parameter
    llm = ZhipuAI(
        model="glm-5.3", api_key="key", base_url="https://api.z.ai/api/paas/v4"
    )
    assert llm.base_url == "https://api.z.ai/api/paas/v4"

    # From BIGMODEL_API_BASE
    with mock.patch.dict(
        os.environ, {"BIGMODEL_API_BASE": "https://custom.bigmodel.cn/api"}, clear=True
    ):
        llm = ZhipuAI(model="glm-5.3", api_key="key")
        assert llm.base_url == "https://custom.bigmodel.cn/api"


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
