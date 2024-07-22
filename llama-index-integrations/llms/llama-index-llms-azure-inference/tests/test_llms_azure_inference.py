import os
import pytest
import json
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core.llms import ChatMessage, MessageRole


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
        temperature=1.0,
        presence_penalty=0.0,
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
        temperature=1.0,
        presence_penalty=0.0,
    )

    assert response.message.role == MessageRole.ASSISTANT
    assert (
        json.loads(response.message.content.strip()).get("message")
        == "Yes, this is a test."
    )
