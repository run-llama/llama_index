from decimal import Decimal
import httpx
from llama_index.core.base.llms.base import BaseLLM
import pytest
import respx
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
from llama_index.llms.litellm import LiteLLM
from llama_index.core.tools import FunctionTool


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LiteLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_chat_response(respx_mock)
    message = ChatMessage(role="user", content="Hey! how's it going?")
    chat_response = llm.chat([message])
    assert chat_response.message.blocks[0].text == "Hello, world!"


@pytest.mark.asyncio()
async def test_achat(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_chat_response(respx_mock)
    message = ChatMessage(role="user", content="Hey! how's it going async?")
    chat_response = await llm.achat([message])
    assert chat_response.message.blocks[0].text == "Hello, world!"

def add(x: Decimal, y: Decimal) -> Decimal:
    return x + y

def test_tool_calling(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_tool_response(respx_mock)
    message = "whats 1+1?"
    chat_response = llm.chat_with_tools(tools=[add_tool], user_msg=message)
    tool_calls = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=True
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"x": 1, "y": 1}


@pytest.mark.asyncio()
async def test_achat_tool_calling(respx_mock: respx.MockRouter, llm: LiteLLM):
    mock_tool_response(respx_mock)
    message = "whats 1+1?"
    chat_response = await llm.achat_with_tools(tools=[add_tool], user_msg=message)
    tool_calls = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=True
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"x": 1, "y": 1}


####################################
## Helper functions  and fixtures ##
####################################

add_tool = FunctionTool.from_defaults(fn=add, name="add")

def mock_chat_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Hello, world!"}}]},
        )
    )


def mock_tool_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Let me calculate that for you.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"x": 1, "y": 1}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )
    )


@pytest.fixture()
def llm():
    return LiteLLM(model="openai/gpt-fake-model")
