import os
import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.zhipuai import ZhipuAI


def test_text_inference_embedding_class():
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


@pytest.mark.skipif(
    os.environ.get("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
def test_completion():
    model = "glm-4"
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    llm = ZhipuAI(model=model, api_key=api_key)
    assert llm.complete("who are you")
