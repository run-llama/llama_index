from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.apertis import Apertis


def test_llm_class():
    """Test that Apertis is a proper LLM subclass."""
    names_of_base_classes = [b.__name__ for b in Apertis.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_class_name():
    """Test the class name method."""
    assert Apertis.class_name() == "Apertis_LLM"


def test_default_model():
    """Test default model initialization."""
    llm = Apertis(api_key="test-key")
    assert llm.model == "gpt-5.2"


def test_custom_model():
    """Test custom model initialization."""
    llm = Apertis(api_key="test-key", model="claude-sonnet-4.5")
    assert llm.model == "claude-sonnet-4.5"


def test_api_base():
    """Test default API base URL."""
    llm = Apertis(api_key="test-key")
    assert llm.api_base == "https://api.apertis.ai/v1"


def test_custom_api_base():
    """Test custom API base URL."""
    custom_base = "https://custom.api.example.com/v1"
    llm = Apertis(api_key="test-key", api_base=custom_base)
    assert llm.api_base == custom_base


def test_is_chat_model():
    """Test that Apertis is configured as a chat model by default."""
    llm = Apertis(api_key="test-key")
    assert llm.is_chat_model is True
