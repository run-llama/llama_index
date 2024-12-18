from unittest.mock import Mock, patch
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.core.base.llms.types import CompletionResponse


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in AnthropicMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes


def test_init():
    m = AnthropicMultiModal(max_tokens=400)
    assert m.max_tokens == 400


def test_tool_response():
    """Test handling of tool responses."""
    llm = AnthropicMultiModal(max_tokens=400)

    # Create mock response with tool input
    mock_content = Mock()
    mock_content.input = {
        "booking_number": "123",
        "carrier": "Test Carrier",
        "total_amount": 1000.0,
    }
    mock_response = Mock()
    mock_response.content = [mock_content]

    with patch.object(llm._client.messages, "create", return_value=mock_response):
        response = llm.complete(
            prompt="test prompt",
            image_documents=[],
            tools=[
                {
                    "name": "tms_order_payload",
                    "description": "Test tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "booking_number": {"type": "string"},
                            "carrier": {"type": "string"},
                            "total_amount": {"type": "number"},
                        },
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "tms_order_payload"},
        )

        assert isinstance(response, CompletionResponse)
        assert isinstance(response.text, str)
        assert "booking_number" in response.text
        assert "123" in response.text
        assert response.raw == mock_response
