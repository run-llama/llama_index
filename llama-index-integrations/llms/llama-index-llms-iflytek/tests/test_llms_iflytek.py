import os

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.iflytek import IFlytek

model = "generalv3.5"
model_non_function_calling = "lite"
api_key = os.environ.get("IFLYTEK_API_KEY", "")


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in IFlytek.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_iflytek_llm_model_alias():
    llm = IFlytek(model=model, api_key="fake")
    assert llm.model == model


def test_iflytek_default_api_base():
    llm = IFlytek(model=model, api_key="fake")
    assert llm.api_base == "https://spark-api-open.xf-yun.com/v1"


def test_iflytek_llm_metadata():
    llm = IFlytek(model=model, api_key="fake")
    assert llm.metadata.is_function_calling_model is True
    llm = IFlytek(model="4.0Ultra", api_key="fake")
    assert llm.metadata.is_function_calling_model is True
    llm = IFlytek(model=model_non_function_calling, api_key="fake")
    assert llm.metadata.is_function_calling_model is False


def test_iflytek_api_key_from_env(monkeypatch):
    monkeypatch.setenv("IFLYTEK_API_KEY", "env-secret")
    llm = IFlytek(model=model)
    assert llm.api_key == "env-secret"


@pytest.mark.skipif(not api_key, reason="No iFlytek API key set")
def test_completion():
    llm = IFlytek(model=model, api_key=api_key)
    response = llm.complete("who are you")
    assert response


@pytest.mark.skipif(not api_key, reason="No iFlytek API key set")
@pytest.mark.asyncio
async def test_async_completion():
    llm = IFlytek(model=model, api_key=api_key)
    response = await llm.acomplete("who are you")
    assert response
