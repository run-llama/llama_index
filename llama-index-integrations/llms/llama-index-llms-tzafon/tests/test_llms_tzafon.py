from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.tzafon import Tzafon


def test_embedding_class():
    """Test that Tzafon inherits from BaseLLM."""
    names_of_base_classes = [b.__name__ for b in Tzafon.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_default_model():
    """Test that Tzafon has the correct default model."""
    # Create instance with a dummy API key to avoid env var requirement
    llm = Tzafon(api_key="test_key")
    assert llm.model == "tzafon.sm-1"


def test_class_name():
    """Test that class_name returns correct value."""
    assert Tzafon.class_name() == "Tzafon"


def test_api_base():
    """Test that Tzafon has the correct default API base."""
    llm = Tzafon(api_key="test_key")
    assert llm.api_base == "https://api.tzafon.ai/v1"


def test_is_chat_model():
    """Test that Tzafon is configured as a chat model by default."""
    llm = Tzafon(api_key="test_key")
    assert llm.metadata.is_chat_model is True


def test_is_not_function_calling_by_default():
    """Test that function calling is disabled by default."""
    llm = Tzafon(api_key="test_key")
    assert llm.metadata.is_function_calling_model is False
