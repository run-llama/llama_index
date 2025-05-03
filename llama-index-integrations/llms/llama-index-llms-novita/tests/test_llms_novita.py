import os
from datetime import datetime

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.novita import NovitaAI

model = "meta-llama/llama-3.1-8b-instruct"
model_function_calling = "deepseek/deepseek_v3"
api_key = os.environ.get("NOVITA_API_KEY", "")


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in NovitaAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_novita_llm_model_alias():
    llm = NovitaAI(model=model, api_key=api_key)
    assert llm.model == model


def test_novita_llm_metadata():
    llm = NovitaAI(model=model_function_calling, api_key=api_key)
    assert llm.metadata.is_function_calling_model is True
    llm = NovitaAI(model=model, api_key=api_key)
    assert llm.metadata.is_function_calling_model is False


@pytest.mark.skipif(not api_key, reason="No Novita API key set")
def test_completion():
    llm = NovitaAI(model=model, api_key=api_key)
    response = llm.complete("who are you")
    print(response)
    assert response


@pytest.mark.skipif(not api_key, reason="No Novita API key set")
@pytest.mark.asyncio
async def test_async_completion():
    llm = NovitaAI(model=model, api_key=api_key)
    response = await llm.acomplete("who are you")
    print(response)
    assert response


@pytest.mark.skipif(not api_key, reason="No Novita API key set")
def test_stream_complete():
    llm = NovitaAI(model=model, api_key=api_key)
    response = llm.stream_complete("who are you")
    responses = []
    for r in response:
        responses.append(r)
        print(r.delta, end="")
    assert responses
    assert len(responses) > 0


@pytest.mark.skipif(not api_key, reason="No Novita API key set")
@pytest.mark.asyncio
async def test_astream_complete():
    llm = NovitaAI(model=model, api_key=api_key)
    response = await llm.astream_complete("who are you")
    responses = []
    async for r in response:
        responses.append(r)
        print(r.delta, end="")
    assert responses
    assert len(responses) > 0


@pytest.mark.skipif(not api_key, reason="No Novita API key set")
def test_function_calling():
    def get_current_time() -> dict:
        """Get the current time."""
        return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    llm = NovitaAI(model=model_function_calling, api_key=api_key)
    tool = FunctionTool.from_defaults(fn=get_current_time)
    response = llm.predict_and_call([tool], "What is the current time?")
    print(response)
    assert response
