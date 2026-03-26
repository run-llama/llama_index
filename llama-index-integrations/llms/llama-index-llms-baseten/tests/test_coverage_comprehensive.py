#!/usr/bin/env python3
"""
Comprehensive test coverage for Baseten dynamic validation functions.
This file ensures all lines in utils.py and base.py are covered.
"""

import sys
import warnings
from unittest.mock import Mock, patch

# Add the Baseten LLM integration to the path
sys.path.insert(
    0,
    "/Users/alexker/code/llama_index/llama-index-integrations/llms/llama-index-llms-baseten",
)

from llama_index.llms.baseten.utils import (
    Model,
    validate_model_dynamic,
    get_available_models_dynamic,
    validate_model_slug,
    SUPPORTED_MODEL_SLUGS,
)
from llama_index.llms.baseten.base import Baseten


def test_model_class():
    """Test the Model class comprehensively."""
    print("Testing Model class...")

    # Test basic creation
    model = Model(id="test-model")
    assert model.id == "test-model"
    assert model.model_type == "chat"
    assert model.client == "Baseten"

    # Test with custom values
    model2 = Model(id="custom-model", model_type="completion", client="Custom")
    assert model2.id == "custom-model"
    assert model2.model_type == "completion"
    assert model2.client == "Custom"

    # Test hash functionality
    model3 = Model(id="test-model")
    assert hash(model) == hash(model3)

    # Test that models can be used in sets
    model_set = {model, model2, model3}
    assert len(model_set) == 2  # model and model3 are the same

    print("✅ Model class tests passed")


def test_get_available_models_dynamic():
    """Test the get_available_models_dynamic function comprehensively."""
    print("Testing get_available_models_dynamic...")

    # Test successful API call
    mock_client = Mock()
    mock_model1 = Mock()
    mock_model1.id = "model-1"
    mock_model2 = Mock()
    mock_model2.id = "model-2"

    mock_response = Mock()
    mock_response.data = [mock_model1, mock_model2]
    mock_client.models.list.return_value = mock_response

    result = get_available_models_dynamic(mock_client)

    assert len(result) == 2
    assert result[0].id == "model-1"
    assert result[1].id == "model-2"
    assert all(isinstance(model, Model) for model in result)

    # Test API call failure fallback
    mock_client.models.list.side_effect = Exception("API Error")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_available_models_dynamic(mock_client)

        assert len(w) == 1
        assert "Failed to fetch models dynamically" in str(w[0].message)

    # Should return static models
    assert len(result) == len(SUPPORTED_MODEL_SLUGS)
    assert all(isinstance(model, Model) for model in result)
    assert result[0].id == SUPPORTED_MODEL_SLUGS[0]

    # Test empty response
    mock_client.models.list.side_effect = None
    mock_response.data = []
    mock_client.models.list.return_value = mock_response

    result = get_available_models_dynamic(mock_client)
    assert len(result) == 0

    print("✅ get_available_models_dynamic tests passed")


def test_validate_model_dynamic():
    """Test the validate_model_dynamic function comprehensively."""
    print("Testing validate_model_dynamic...")

    # Test valid model success
    mock_client = Mock()
    mock_model = Mock()
    mock_model.id = "valid-model"

    mock_response = Mock()
    mock_response.data = [mock_model]
    mock_client.models.list.return_value = mock_response

    # Should not raise any exception
    validate_model_dynamic(mock_client, "valid-model")

    # Test invalid model with suggestions
    mock_model1 = Mock()
    mock_model1.id = "deepseek-model"
    mock_model2 = Mock()
    mock_model2.id = "llama-model"

    mock_response.data = [mock_model1, mock_model2]
    mock_client.models.list.return_value = mock_response

    try:
        validate_model_dynamic(mock_client, "deepseek")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "not found in available models" in error_msg
        assert "Did you mean" in error_msg

    # Test invalid model without suggestions
    mock_model3 = Mock()
    mock_model3.id = "completely-different-model"
    mock_response.data = [mock_model3]
    mock_client.models.list.return_value = mock_response

    try:
        validate_model_dynamic(mock_client, "totally-unrelated-model")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "not found in available models" in error_msg
        assert "Available models" in error_msg

    # Test API failure fallback to static validation
    mock_client.models.list.side_effect = Exception("Network error")

    # Use a valid static model
    valid_static_model = SUPPORTED_MODEL_SLUGS[0]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_model_dynamic(mock_client, valid_static_model)

        assert len(w) == 1
        warning_msg = str(w[0].message)
        assert "Failed to fetch models dynamically" in warning_msg

    # Test API failure with invalid static model
    try:
        validate_model_dynamic(mock_client, "invalid-static-model")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # The error message comes from dynamic validation, not static
        assert "not found in available models" in error_msg

    # Test validation error re-raise
    mock_client.models.list.side_effect = ValueError(
        "Model not found in available models"
    )

    try:
        validate_model_dynamic(mock_client, "some-model")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        # The error gets re-raised with a different message format
        assert "not found in available models" in str(e)

    print("✅ validate_model_dynamic tests passed")


def test_baseten_class():
    """Test the Baseten class dynamic functionality."""
    print("Testing Baseten class...")

    # Test available_models property with model_apis=True
    with patch(
        "llama_index.llms.baseten.base.get_available_models_dynamic"
    ) as mock_get_models:
        with patch(
            "llama_index.llms.baseten.base.validate_model_dynamic"
        ) as mock_validate:
            with patch(
                "llama_index.llms.baseten.base.get_from_param_or_env"
            ) as mock_get_key:
                mock_get_key.return_value = "fake-api-key"
                mock_models = [Model(id="model-1"), Model(id="model-2")]
                mock_get_models.return_value = mock_models

                llm = Baseten(model_id="test-model", model_apis=True)
                llm._get_client = Mock()

                result = llm.available_models

                assert result == mock_models
                mock_get_models.assert_called_once()

    # Test available_models property with model_apis=False
    with patch("llama_index.llms.baseten.base.get_from_param_or_env") as mock_get_key:
        mock_get_key.return_value = "fake-api-key"
        llm = Baseten(model_id="test-model", model_apis=False)
        result = llm.available_models
        assert len(result) == 1
        assert result[0].id == "test-model"

    # Test available_models property with dedicated deployment but no model attribute
    with patch("llama_index.llms.baseten.base.get_from_param_or_env") as mock_get_key:
        mock_get_key.return_value = "fake-api-key"
        llm = Baseten(model_id="test-model", model_apis=False)
        delattr(llm, "model")
        result = llm.available_models
        assert result == []

    # Test dynamic validation in constructor
    with patch("llama_index.llms.baseten.base.validate_model_dynamic") as mock_validate:
        with patch("openai.OpenAI") as mock_client_class:
            with patch(
                "llama_index.llms.baseten.base.get_from_param_or_env"
            ) as mock_get_key:
                mock_get_key.return_value = "fake-api-key"
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                llm = Baseten(model_id="test-model", model_apis=True)
                mock_validate.assert_called_once_with(mock_client, "test-model")

    # Test no validation for dedicated deployment
    with patch("llama_index.llms.baseten.base.validate_model_dynamic") as mock_validate:
        with patch(
            "llama_index.llms.baseten.base.get_from_param_or_env"
        ) as mock_get_key:
            mock_get_key.return_value = "fake-api-key"
            llm = Baseten(model_id="test-model", model_apis=False)
            mock_validate.assert_not_called()

    print("✅ Baseten class tests passed")


def test_static_functions():
    """Test static utility functions."""
    print("Testing static functions...")

    # Test validate_model_slug with valid model
    valid_model = SUPPORTED_MODEL_SLUGS[0]
    validate_model_slug(valid_model)  # Should not raise exception

    # Test validate_model_slug with invalid model
    try:
        validate_model_slug("invalid-model")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "not supported by Baseten Model APIs" in error_msg
        assert "Supported models are:" in error_msg

    print("✅ Static functions tests passed")


def main():
    """Run all tests."""
    print("🧪 Running Comprehensive Coverage Tests")
    print("=" * 50)

    try:
        test_model_class()
        test_get_available_models_dynamic()
        test_validate_model_dynamic()
        test_baseten_class()
        test_static_functions()

        print("=" * 50)
        print("🎉 All coverage tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
