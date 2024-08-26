"""LlamaIndex toolkit."""

from typing import List

from llama_index.core.bridge.langchain import BaseTool, BaseToolkit
from llama_index.core.bridge.pydantic import Field
from llama_index.core.langchain_helpers.agents.tools import (
    IndexToolConfig,
    LlamaIndexTool,
)


class LlamaToolkit(BaseToolkit):
    """Toolkit for interacting with Llama indices."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index_configs: List[IndexToolConfig] = Field(default_factory=list)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        index_tools: List[BaseTool] = [
            LlamaIndexTool.from_tool_config(tool_config=tool_config)
            for tool_config in self.index_configs
        ]

        return index_tools
