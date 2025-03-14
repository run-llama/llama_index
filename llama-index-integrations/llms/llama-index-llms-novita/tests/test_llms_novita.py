import json
import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.novita import NovitaAI

model = "meta-llama/llama-3.1-8b-instruct"
model_function_calling = "deepseek/deepseek_v3"
api_key = "you api key"

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

def test_novita_available_models():
    llm = NovitaAI(model=model, api_key=api_key)
    model_list = llm.available_models
    for m in model_list:
        print(m)
    assert model_list

def test_novita_retrieve_model():
    llm = NovitaAI(model=model, api_key=api_key)
    m = llm.retrieve_model
    print(m)
    assert m

def test_completion():
    llm = NovitaAI(model=model, api_key=api_key)
    response = llm.complete("who are you")
    print(response)
    assert response

@pytest.mark.asyncio()
async def test_async_completion():
    llm = NovitaAI(model=model, api_key=api_key)
    response = await llm.acomplete("who are you")
    print(response)
    assert response


def test_stream_complete():
    llm = NovitaAI(model=model, api_key=api_key)
    response = llm.stream_complete("who are you")
    responses = []
    for r in response:
        responses.append(r)
        print(r.delta, end="")
    assert responses
    assert len(responses) > 0

@pytest.mark.asyncio()
async def test_astream_complete():
    llm = NovitaAI(model=model, api_key=api_key)
    response = await llm.astream_complete("who are you")
    responses = []
    async for r in response:
        responses.append(r)
        print(r.delta, end="")
    assert responses
    assert len(responses) > 0

def test_function_calling():
    def get_weather(location):
        return json.dumps({"location": location, "temperature": "60 degrees Fahrenheit"})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]
    llm = NovitaAI(model=model_function_calling, api_key=api_key)
    response = llm.complete(
        "What is the weather in San Francisco?",
        tools = tools,
        tool_choice = "auto")
    func_call_list = llm.get_tool_calls_from_response(response)
    for func_call in func_call_list:
        if func_call.tool_name == "get_weather":
            print(get_weather(location=func_call.tool_kwargs.get("location")))
    assert response

