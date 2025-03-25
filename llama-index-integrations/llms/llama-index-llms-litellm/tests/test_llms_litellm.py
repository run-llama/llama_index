import httpx
from llama_index.core.base.llms.base import BaseLLM
import pytest
import respx
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
from llama_index.llms.litellm import LiteLLM


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LiteLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def mock_llm_response(respx_mock: respx.MockRouter):
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Hello, world!"}}]},
        )
    )


def test_chat(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    mock_llm_response(respx_mock)
    # Define a chat message
    message = ChatMessage(role="user", content="Hey! how's it going?")

    # Initialize LiteLLM with the desired model
    llm = LiteLLM(model="openai/gpt-fake-model")

    # Call the chat method with the message
    chat_response = llm.chat([message])

    # Print the response
    assert chat_response.message.blocks[0].text == "Hello, world!"

@pytest.mark.asyncio
async def test_achat(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    mock_llm_response(respx_mock)
    # Define a chat message
    message = ChatMessage(role="user", content="Hey! how's it going async?")

    # Initialize LiteLLM with the desired model
    llm = LiteLLM(model="openai/gpt-fake-model")

    # Call the chat method with the message
    chat_response = await llm.achat([message])

    # Print the response
    assert chat_response.message.blocks[0].text == "Hello, world!"


def test_chat(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    mock_llm_response(respx_mock)
    # Define a chat message
    message = ChatMessage(role="user", content="Hey! how's it going?")

    # Initialize LiteLLM with the desired model
    llm = LiteLLM(model="openai/gpt-fake-model")

    # Call the chat method with the message
    chat_response = llm.astream_chat_with_tools([message])

    # Print the response
    assert chat_response.message.blocks[0].text == "Hello, world!"