from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type, Dict
from pydantic import BaseModel
from langchain.tools import Tool, StructuredTool


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = None


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> None:
        pass

    def _process_langchain_tool_kwargs(
        self,
        langchain_tool_kwargs: Any,
    ) -> Dict[str, Any]:
        """Process langchain tool kwargs."""
        if "name" not in langchain_tool_kwargs:
            langchain_tool_kwargs["name"] = self.metadata.name or ""
        if "description" not in langchain_tool_kwargs:
            langchain_tool_kwargs["description"] = self.metadata.description
        if "fn_schema" not in langchain_tool_kwargs:
            langchain_tool_kwargs["args_schema"] = self.metadata.fn_schema
        return langchain_tool_kwargs

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return Tool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> StructuredTool:
        """To langchain structured tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return StructuredTool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )
