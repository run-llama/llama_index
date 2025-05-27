from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from unittest.mock import patch
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceInferenceAPI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required is correctly passed to the API request when True."""
    with (
        patch("huggingface_hub.InferenceClient"),
        patch("huggingface_hub.AsyncInferenceClient"),
    ):
        llm = HuggingFaceInferenceAPI(model_name="model_name")

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert result["tool_choice"] == "required"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    with (
        patch("huggingface_hub.InferenceClient"),
        patch("huggingface_hub.AsyncInferenceClient"),
    ):
        llm = HuggingFaceInferenceAPI(model_name="model_name")

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
    )

    assert result["tool_choice"] == "auto"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"
