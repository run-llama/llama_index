from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from llama_index.bridge.langchain import StructuredTool, Tool
from pydantic import BaseModel


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        return self.fn_schema.schema()["properties"]

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    def to_openai_function(self) -> Dict[str, Any]:
        """To OpenAI function."""
        if self.fn_schema is None:
            parameters = {
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
                "type": "object",
            }
        else:
            parameters = self.fn_schema.schema()

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> Any:
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
