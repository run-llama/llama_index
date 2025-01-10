"""Test Bedrock multi-modal LLM."""
import pytest
from unittest.mock import patch

from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.bedrock import BedrockMultiModal
from llama_index.core.schema import ImageDocument


def test_class_name():
    """Test class name."""
    llm = BedrockMultiModal()
    assert llm.class_name() == "bedrock_multi_modal_llm"


def test_init():
    """Test initialization."""
    llm = BedrockMultiModal(max_tokens=400)
    assert llm.max_tokens == 400
    assert llm.model == "anthropic.claude-3-sonnet-20240229-v1:0"


def test_inheritance():
    """Test inheritance."""
    assert issubclass(BedrockMultiModal, MultiModalLLM)


def test_model_validation():
    """Test model validation."""
    with pytest.raises(ValueError, match="Invalid model"):
        BedrockMultiModal(model="invalid-model")


@patch("boto3.Session")
def test_completion(mock_session):
    """Test completion."""
    # Mock the invoke_model response
    mock_client = mock_session.return_value.client.return_value
    mock_client.invoke_model.return_value = {"content": [{"text": "test response"}]}

    llm = BedrockMultiModal()
    image_doc = ImageDocument(image="base64_encoded_string")

    response = llm.complete(prompt="test prompt", image_documents=[image_doc])

    assert response.text == "test response"
    # Verify the call was made with correct parameters
    mock_client.invoke_model.assert_called_once()
    call_args = mock_client.invoke_model.call_args[1]
    assert "modelId" in call_args
    assert call_args["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"


@pytest.mark.asyncio()
@patch("aioboto3.Session")
async def test_async_completion(mock_session):
    """Test async completion."""
    # Mock the async client
    mock_client = mock_session.return_value.client.return_value
    mock_client.__aenter__.return_value.invoke_model.return_value = {
        "content": [{"text": "async test response"}]
    }

    llm = BedrockMultiModal()
    image_doc = ImageDocument(image="base64_encoded_string")

    response = await llm.acomplete(prompt="test prompt", image_documents=[image_doc])

    assert response.text == "async test response"
    # No need to verify call args for async as the mock is structured differently
