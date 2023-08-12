"""LlamaIndex toolkit."""

from typing import List

from llama_index.bridge.langchain import BaseTool, BaseToolkit
from llama_index.tools import AsyncBaseTool, adapt_to_async_tool
from pydantic import Field

from llama_index.langchain_helpers.agents.tools import (
    IndexToolConfig,
    LlamaIndexTool,
)


class LlamaToolkit(BaseToolkit):
    """Toolkit for interacting with Llama indices."""

    index_configs: List[IndexToolConfig] = Field(default_factory=list)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[AsyncBaseTool]:
        """Get the tools in the toolkit."""
        index_tools: List[BaseTool] = [
            adapt_to_async_tool(
                LlamaIndexTool.from_tool_config(tool_config=tool_config)
            )
            for tool_config in self.index_configs
        ]

        return index_tools
