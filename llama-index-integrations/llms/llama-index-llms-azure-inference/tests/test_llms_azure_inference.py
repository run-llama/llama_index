import asyncio
import json
import logging
import os
from unittest import mock

# import aiohttp to force Pants to include it in the required dependencies
import aiohttp  # noqa
import pytest
from azure.ai.inference.models import (
    ChatChoice,
    ChatCompletions,
    ChatResponseMessage,
    ModelInfo,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_inference import AzureAICompletionsModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def test_params() -> dict:
    return {
        "messages": [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. When you are asked about if this "
                "is a test, you always reply 'Yes, this is a test.'",
            ),
            ChatMessage(role="user", content="Is this a test?"),
        ],
        "top_p": 1.0,
        "temperature": 0.0,
    }


@pytest.fixture()
def test_llm():
    with mock.patch(
        "llama_index.llms.azure_inference.base.ChatCompletionsClient", autospec=True
    ):
        with mock.patch(
            "llama_index.llms.azure_inference.base.ChatCompletionsClientAsync",
            autospec=True,
        ):
            llm = AzureAICompletionsModel(
                endpoint="https://my-endpoint.inference.ai.azure.com",
                credential="my-api-key",
            )
    llm._client.complete.return_value = ChatCompletions(
        choices=[
            ChatChoice(
                message=ChatResponseMessage(
                    content="Yes, this is a test.", role="assistant"
                )
            )
        ]
    )
    llm._client.get_model_info.return_value = ModelInfo(
        model_name="my_model_name",
        model_provider_name="my_provider_name",
        model_type="chat-completions",
    )
    llm._async_client.complete = mock.AsyncMock(
        return_value=ChatCompletions(
            choices=[
                ChatChoice(
                    message=ChatResponseMessage(
                        content="Yes, this is a test.", role="assistant"
                    )
                )
            ]
        )
    )
    return llm


@pytest.fixture()
def test_llm_json():
    with mock.patch(
        "llama_index.llms.azure_inference.base.ChatCompletionsClient", autospec=True
    ):
        llm = AzureAICompletionsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
        )
    llm._client.complete.return_value = ChatCompletions(
        choices=[
            ChatChoice(
                message=ChatResponseMessage(
                    content='{ "message": "Yes, this is a test." }', role="assistant"
                )
            )
        ]
    )
    return llm


@pytest.fixture()
def test_llm_tools():
    with mock.patch(
        "llama_index.llms.azure_inference.base.ChatCompletionsClient", autospec=True
    ):
        llm = AzureAICompletionsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
        )
    llm._client.complete.return_value = ChatCompletions(
        choices=[
            ChatChoice(
                message=ChatResponseMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "abc0dF1gh",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": None,
                                "call_id": None,
                            },
                        }
                    ],
                )
            )
        ]
    )
    return llm


def test_chat_completion(test_llm: AzureAICompletionsModel, test_params: dict):
    """Tests the basic chat completion functionality."""
    response = test_llm.chat(**test_params)

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content.strip() == "Yes, this is a test."


def test_achat_completion(
    test_llm: AzureAICompletionsModel,
    loop: asyncio.AbstractEventLoop,
    test_params: dict,
):
    """Tests the basic chat completion functionality asynchronously."""
    response = loop.run_until_complete(test_llm.achat(**test_params))

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content.strip() == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_stream_chat_completion(test_params: dict):
    """Tests the basic chat completion functionality with streaming."""
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(model_name=model_name)

    response_stream = llm.stream_chat(**test_params)

    buffer = ""
    for chunk in response_stream:
        buffer += chunk.delta

    assert buffer.strip() == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_astream_chat_completion(test_params: dict, loop: asyncio.AbstractEventLoop):
    """Tests the basic chat completion functionality with streaming."""
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(model_name=model_name)

    async def iterate():
        stream = await llm.astream_chat(**test_params)
        buffer = ""
        async for chunk in stream:
            buffer += chunk.delta

        return buffer

    response = loop.run_until_complete(iterate())
    assert response.strip() == "Yes, this is a test."


def test_chat_completion_kwargs(
    test_llm_json: AzureAICompletionsModel,
):
    """Tests chat completions using extra parameters."""
    test_llm_json.model_kwargs.update({"response_format": {"type": "json_object"}})
    response = test_llm_json.chat(
        [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. When you are asked about if this "
                "is a test, you always reply 'Yes, this is a test.' in a JSON object with "
                "key 'message'.",
            ),
            ChatMessage(role="user", content="Is this a test?"),
        ],
        temperature=0.0,
        top_p=1.0,
    )

    assert response.message.role == MessageRole.ASSISTANT
    assert (
        json.loads(response.message.content.strip()).get("message")
        == "Yes, this is a test."
    )


def test_chat_completion_with_tools(test_llm_tools: AzureAICompletionsModel):
    """Tests the chat completion functionality with the help of tools."""

    def echo(message: str) -> str:
        """Echoes the user's message."""
        print("Echo: " + message)
        return message

    response = test_llm_tools.chat_with_tools(
        user_msg="Is this a test?",
        chat_history=[
            ChatMessage(
                role="system",
                content="You are an assistant that always echoes the user's message. To echo a message, use the 'Echo' tool.",
            ),
        ],
        tools=[
            FunctionTool.from_defaults(
                fn=echo,
                name="echo",
                description="Echoes the user's message.",
            ),
        ],
        verbose=True,
    )

    assert response.message.role == MessageRole.ASSISTANT
    assert len(response.message.additional_kwargs["tool_calls"]) == 1
    assert (
        response.message.additional_kwargs["tool_calls"][0]["function"]["name"]
        == "echo"
    )


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_chat_completion_gpt4o_api_version(test_params: dict):
    """Test chat completions endpoint with api_version indicated for a GPT model."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", "gpt-4o")

    llm = AzureAICompletionsModel(
        model_name=model_name, api_version="2024-05-01-preview"
    )

    response = llm.chat(**test_params)

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content.strip() == "Yes, this is a test."


def test_get_metadata(test_llm: AzureAICompletionsModel, caplog):
    """Tests if we can get model metadata back from the endpoint. If so,
    model_name should not be 'unknown'. Some endpoints may not support this
    and in those cases a warning should be logged.
    """
    response = test_llm.metadata

    assert (
        response.model_name != "unknown"
        or "does not support model metadata retrieval" in caplog.text
    )
