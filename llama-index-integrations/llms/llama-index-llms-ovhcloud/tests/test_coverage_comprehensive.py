#!/usr/bin/env python3
"""
Comprehensive test coverage for OVHcloud AI Endpoints dynamic validation functions.
This file ensures all lines in utils.py and base.py are covered.
"""

import sys
from unittest.mock import Mock, patch
from llama_index.llms.ovhcloud.utils import Model
from llama_index.llms.ovhcloud.base import OVHcloud


def test_model_class():
    """Test the Model class comprehensively."""
    print("Testing Model class...")

    # Test basic creation
    model = Model(id="test-model")
    assert model.id == "test-model"
    assert model.model_type == "chat"
    assert model.client == "OVHcloud"

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


def test_ovhcloud_class():
    """Test the OVHcloud class dynamic functionality."""
    print("Testing OVHcloud class...")

    # Test available_models property with exception (falls back to current model)
    llm = OVHcloud(model="test-model", api_key="fake-api-key")
    result = llm.available_models
    assert len(result) == 1
    assert result[0].id == "test-model"

    # Test constructor with API key
    with patch("openai.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        llm = OVHcloud(model="test-model", api_key="fake-api-key")

    # Test constructor with empty API key
    llm = OVHcloud(model="test-model", api_key="")

    print("✅ OVHcloud class tests passed")


def main():
    """Run all tests."""
    print("🧪 Running Comprehensive Coverage Tests")
    print("=" * 50)

    try:
        test_model_class()
        test_ovhcloud_class()

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
