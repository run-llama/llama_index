import logging
import os
import pytest
import json
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_chat_completion():
    """Tests the basic chat completion functionality."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(model_name=model_name)

    response = llm.chat(
        [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. When you are asked about if this "
                "is a test, you always reply 'Yes, this is a test.'",
            ),
            ChatMessage(role="user", content="Is this a test?"),
        ],
        temperature=0.0,
        top_p=1.0,
    )

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content.strip() == "Yes, this is a test."


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_chat_completion_kwargs():
    """Tests chat completions using extra parameters."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(
        model_name=model_name,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    response = llm.chat(
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


@pytest.mark.skipif(
    not {
        "AZURE_INFERENCE_ENDPOINT",
        "AZURE_INFERENCE_CREDENTIAL",
    }.issubset(set(os.environ)),
    reason="Azure AI endpoint and/or credential are not set.",
)
def test_chat_completion_with_tools():
    """Tests the chat completion functionality with the help of tools."""
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(model_name=model_name)

    def echo(message: str) -> str:
        """Echoes the user's message."""
        print("Echo: " + message)
        return message

    response = llm.chat_with_tools(
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
def test_get_metadata(caplog):
    """Tests if we can get model metadata back from the endpoint. If so,
    model_name should not be 'unknown'. Some endpoints may not support this
    and in those cases a warning should be logged.
    """
    # In case the endpoint being tested serves more than one model
    model_name = os.environ.get("AZURE_INFERENCE_MODEL", None)

    llm = AzureAICompletionsModel(model_name=model_name)

    response = llm.metadata

    assert (
        response.model_name != "unknown"
        or "does not support model metadata retrieval" in caplog.text
    )
    assert not model_name or response.model_name == model_name
