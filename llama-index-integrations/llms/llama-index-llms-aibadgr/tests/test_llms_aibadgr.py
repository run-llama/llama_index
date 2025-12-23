from llama_index.llms.aibadgr import AIBadgr


def test_class_name():
    """Test that class name is correct."""
    names_of_base_classes = [b.__name__ for b in AIBadgr.__mro__]
    assert AIBadgr.class_name() == "AIBadgr"
    assert "OpenAILike" in names_of_base_classes


def test_initialization():
    """Test that AIBadgr can be initialized with default and custom parameters."""
    # Test with premium model
    llm = AIBadgr(model="premium", api_key="test_key")
    assert llm.model == "premium"
    assert llm.api_key == "test_key"
    assert llm.api_base == "https://aibadgr.com/api/v1"

    # Test with custom model
    llm = AIBadgr(model="basic", api_key="test_key")
    assert llm.model == "basic"

    # Test with custom base URL
    llm = AIBadgr(
        model="premium", api_key="test_key", api_base="https://custom.url/api/v1"
    )
    assert llm.api_base == "https://custom.url/api/v1"
