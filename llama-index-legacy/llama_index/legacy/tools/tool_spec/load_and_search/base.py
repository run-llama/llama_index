"""Ad-hoc data loader tool.

Tool that wraps any data loader, and is able to load data on-demand.

"""

from typing import Any, Dict, List, Optional, Type

from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.indices.base import BaseIndex
from llama_index.legacy.indices.vector_store import VectorStoreIndex
from llama_index.legacy.tools.function_tool import FunctionTool
from llama_index.legacy.tools.tool_spec.base import SPEC_FUNCTION_TYPE, BaseToolSpec
from llama_index.legacy.tools.types import ToolMetadata
from llama_index.legacy.tools.utils import create_schema_from_function


class LoadAndSearchToolSpec(BaseToolSpec):
    """Load and Search Tool.

    This tool can be used with other tools that load large amounts of
    information. Compared to OndemandLoaderTool this returns two tools,
    one to retrieve data to an index and another to allow the Agent to search
    the retrieved data with a natural language query string.

    """

    loader_prompt = """
        Use this tool to load data from the following function. It must then be read from
        the corresponding read_{} function.

        {}
    """

    # TODO, more general read prompt, not always natural language?
    reader_prompt = """
        Once data has been loaded from {} it can then be read using a natural
        language query from this function.

        You are required to pass the natural language query argument when calling this endpoint

        Args:
            query (str): The natural language query used to retreieve information from the index
    """

    def __init__(
        self,
        tool: FunctionTool,
        index_cls: Type[BaseIndex],
        index_kwargs: Dict,
        metadata: ToolMetadata,
        index: Optional[BaseIndex] = None,
    ) -> None:
        """Init params."""
        self._index_cls = index_cls
        self._index_kwargs = index_kwargs
        self._index = index
        self._metadata = metadata
        self._tool = tool

        if self._metadata.name is None:
            raise ValueError("Tool name cannot be None")
        self.spec_functions = [
            self._metadata.name,
            f"read_{self._metadata.name}",
        ]
        self._tool_list = [
            FunctionTool.from_defaults(
                fn=self.load,
                name=self._metadata.name,
                description=self.loader_prompt.format(
                    self._metadata.name, self._metadata.description
                ),
                fn_schema=self._metadata.fn_schema,
            ),
            FunctionTool.from_defaults(
                fn=self.read,
                name=str(f"read_{self._metadata.name}"),
                description=self.reader_prompt.format(metadata.name),
                fn_schema=create_schema_from_function("ReadData", self.read),
            ),
        ]

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @classmethod
    def from_defaults(
        cls,
        tool: FunctionTool,
        index_cls: Optional[Type[BaseIndex]] = None,
        index_kwargs: Optional[Dict] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "LoadAndSearchToolSpec":
        """From defaults."""
        index_cls = index_cls or VectorStoreIndex
        index_kwargs = index_kwargs or {}
        if name is None:
            name = tool.metadata.name
        if description is None:
            description = tool.metadata.description
        if fn_schema is None:
            fn_schema = tool.metadata.fn_schema
        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(
            tool=tool,
            index_cls=index_cls,
            index_kwargs=index_kwargs,
            metadata=metadata,
        )

    def to_tool_list(
        self,
        spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None,
        func_to_metadata_mapping: Optional[Dict[str, ToolMetadata]] = None,
    ) -> List[FunctionTool]:
        return self._tool_list

    def load(self, *args: Any, **kwargs: Any) -> Any:
        # Call the wrapped tool and save the result in the index
        docs = self._tool(*args, **kwargs).raw_output
        if self._index:
            for doc in docs:
                self._index.insert(doc, **self._index_kwargs)
        else:
            self._index = self._index_cls.from_documents(docs, **self._index_kwargs)
        return (
            "Content loaded! You can now search the information using read_{}".format(
                self._metadata.name
            )
        )

    def read(self, query: str) -> Any:
        # Query the index for the result
        if not self._index:
            return (
                "Error: No content has been loaded into the index. "
                f"You must call {self._metadata.name} first"
            )
        query_engine = self._index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
