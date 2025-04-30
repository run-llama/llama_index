import unittest
from unittest.mock import patch, MagicMock
import pytest
from llama_index.llms.text_generation_inference.utils import get_model_name
from llama_index.llms.text_generation_inference.base import TextGenerationInference
from llama_index.core.base.llms.types import LLMMetadata


class TestTGIUtils(unittest.TestCase):
    @patch("llama_index.llms.text_generation_inference.utils.requests.get")
    def test_get_model_name_returns_model_id(self, mock_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2"
        }
        mock_get.return_value = mock_response

        # Call the function
        result = get_model_name("http://localhost:8080")

        # Check the expected URL was called
        mock_get.assert_called_once_with("http://localhost:8080/info")

        # Verify result
        assert result == "mistralai/Mistral-7B-Instruct-v0.2"

    @patch("llama_index.llms.text_generation_inference.utils.requests.get")
    def test_get_model_name_returns_none_when_no_model_id(self, mock_get):
        # Setup mock response without model_id
        mock_response = MagicMock()
        mock_response.json.return_value = {"version": "2.5.0", "max_total_tokens": 4096}
        mock_get.return_value = mock_response

        # Call the function
        result = get_model_name("http://localhost:8080")

        # Check the expected URL was called
        mock_get.assert_called_once_with("http://localhost:8080/info")

        # Verify result is None when model_id is not present
        assert result is None


class TestModelNameValidation(unittest.TestCase):
    def test_llm_metadata_requires_model_name(self):
        """Test that LLMMetadata requires a valid model_name (not None)."""
        # This test verifies the validation requirement that our fix addresses
        with pytest.raises(Exception):
            # Without our fix, this would raise a validation error
            LLMMetadata(
                context_window=4096,
                model_name=None,  # This should cause a validation error
                is_chat_model=True,
            )

    def test_llm_metadata_accepts_valid_model_name(self):
        """Test that LLMMetadata accepts a valid model_name."""
        # Create metadata with valid model_name - should not raise an error
        metadata = LLMMetadata(
            context_window=4096,
            model_name="test-model",  # Valid string model name
            is_chat_model=True,
        )
        assert metadata.model_name == "test-model"


class TestInitializationFlow(unittest.TestCase):
    """Tests specifically for the initialization flow and model_name handling."""

    @patch("llama_index.llms.text_generation_inference.base.get_model_name")
    def test_get_model_name_called_during_init(self, mock_get_model_name):
        """Test that get_model_name is called during initialization with the correct URL."""
        # Setup
        test_url = "http://localhost:8080"
        mock_get_model_name.return_value = "test-model"

        # We need to patch these to prevent actual HTTP calls, but we're not testing them
        with patch("llama_index.llms.text_generation_inference.base.TGIClient"), patch(
            "llama_index.llms.text_generation_inference.base.TGIAsyncClient"
        ), patch(
            "llama_index.llms.text_generation_inference.base.get_max_total_tokens"
        ), patch(
            "llama_index.llms.text_generation_inference.base.resolve_tgi_function_call"
        ), patch.object(
            TextGenerationInference, "_sync_client", new=MagicMock()
        ), patch.object(
            TextGenerationInference, "_async_client", new=MagicMock()
        ):
            # Create the instance - this should call get_model_name
            tgi = TextGenerationInference(model_url=test_url)

            # Verify get_model_name was called with the correct URL
            mock_get_model_name.assert_called_once_with(test_url)

            # Additional verification - model_name should match our mocked return value
            assert tgi.model_name == "test-model"


# Test that the fix resolves the actual issue (validation error)
@patch("llama_index.llms.text_generation_inference.utils.requests.get")
def test_fix_resolves_validation_error(mock_get):
    """Integration test for the fix - verifies that get_model_name prevents validation error."""
    # The fix should prevent validation errors when model_name is None

    # Setup test scenario where API returns a valid model_id
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "version": "2.5.0",
        "max_total_tokens": 4096,
    }
    mock_get.return_value = mock_response

    # Create a metadata object using value from get_model_name
    # This simulates what happens in TextGenerationInference.__init__
    model_name = get_model_name("http://localhost:8080")

    # This would fail without the fix (when model_name was None)
    metadata = LLMMetadata(
        context_window=4096,
        model_name=model_name,  # Now set from API
        is_chat_model=True,
    )

    # Verify model_name is correctly set
    assert metadata.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
