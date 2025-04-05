import os
from unittest.mock import patch, Mock

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.perplexity.base import PerplexityToolSpec

def test_inheritance():
    """Test that PerplexityToolSpec inherits from BaseToolSpec."""
    base_names = [cls.__name__ for cls in PerplexityToolSpec.__mro__]
    assert BaseToolSpec.__name__ in base_names

@patch("llama_index.tools.perplexity.base.requests.request")
def test_chat_completion_with_sonar_pro(mock_request):
    """
    Test that the chat_completion method constructs the payload correctly
    and uses the 'sonar-pro' model when specified.
    """
    # Create a dummy response object with a text attribute.
    dummy_response = Mock()
    dummy_response.text = "dummy response"
    mock_request.return_value = dummy_response

    # Instantiate the tool with a dummy API key.
    dummy_api_key = "dummy_key"
    tool = PerplexityToolSpec(api_key=dummy_api_key)

    # Call the chat_completion method with model set to "sonar-pro".
    query = "What is going on today?"
    response = tool.chat_completion(query, model="sonar-pro")

    # Verify that the response is the dummy response.
    assert response == "dummy response"

    # Verify that requests.request was called with the correct headers and payload.
    args, kwargs = mock_request.call_args

    # Check URL and headers.
    assert kwargs["headers"]["Authorization"] == f"Bearer {dummy_api_key}"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["json"]["model"] == "sonar-pro"
    
    # Verify that the messages list is properly constructed.
    messages = kwargs["json"]["messages"]
    assert any(msg["role"] == "system" for msg in messages)
    assert any(msg["role"] == "user" and msg["content"] == query for msg in messages)
