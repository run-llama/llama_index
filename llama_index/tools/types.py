from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from langchain.tools import Tool, StructuredTool


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> None:
        pass

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        return Tool.from_function(
            fn=self.__call__,
            name=self.metadata.name or "",
            description=self.metadata.description,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> StructuredTool:
        """To langchain structured tool."""
        return StructuredTool.from_function(
            fn=self.__call__,
            name=self.metadata.name or "",
            description=self.metadata.description,
            **langchain_tool_kwargs,
        )
