import os
from datetime import datetime

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.tokenmix import TokenMix

model = "deepseek/deepseek-v4-pro"
non_function_calling_model = "some-vendor/plain-text-model"
api_key = os.environ.get("TOKENMIX_API_KEY", "")


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in TokenMix.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_tokenmix_llm_model_alias():
    llm = TokenMix(model=model, api_key="dummy")
    assert llm.model == model


def test_tokenmix_llm_metadata():
    llm = TokenMix(model=model, api_key="dummy")
    assert llm.metadata.is_function_calling_model is True
    llm = TokenMix(model=non_function_calling_model, api_key="dummy")
    assert llm.metadata.is_function_calling_model is False


def test_tokenmix_default_api_base():
    llm = TokenMix(model=model, api_key="dummy")
    assert llm.api_base == "https://api.tokenmix.ai/v1"


@pytest.mark.skipif(not api_key, reason="No TokenMix API key set")
def test_completion():
    llm = TokenMix(model=model, api_key=api_key)
    response = llm.complete("who are you")
    print(response)
    assert response


@pytest.mark.skipif(not api_key, reason="No TokenMix API key set")
@pytest.mark.asyncio
async def test_async_completion():
    llm = TokenMix(model=model, api_key=api_key)
    response = await llm.acomplete("who are you")
    print(response)
    assert response


@pytest.mark.skipif(not api_key, reason="No TokenMix API key set")
def test_stream_complete():
    llm = TokenMix(model=model, api_key=api_key)
    response = llm.stream_complete("who are you")
    responses = []
    for r in response:
        responses.append(r)
        print(r.delta, end="")
    assert responses
    assert len(responses) > 0


@pytest.mark.skipif(not api_key, reason="No TokenMix API key set")
def test_function_calling():
    def get_current_time() -> dict:
        """Get the current time."""
        return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    llm = TokenMix(model=model, api_key=api_key)
    tool = FunctionTool.from_defaults(fn=get_current_time)
    response = llm.predict_and_call([tool], "What is the current time?")
    print(response)
    assert response
