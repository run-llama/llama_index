import pytest
from llama_index.core.tools.types import ToolMetadata


def test_toolmetadata_openai_tool_description_max_length() -> None:
    openai_tool_description_limit = 1024
    valid_description = "a" * openai_tool_description_limit
    invalid_description = "a" * (1 + openai_tool_description_limit)

    ToolMetadata(valid_description).to_openai_tool()
    ToolMetadata(invalid_description).to_openai_tool(skip_length_check=True)

    with pytest.raises(ValueError):
        ToolMetadata(invalid_description).to_openai_tool()
