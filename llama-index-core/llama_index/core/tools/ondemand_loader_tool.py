"""
Ad-hoc data loader tool.

Tool that wraps any data loader, and is able to load data on-demand.

"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function


class OnDemandLoaderTool(AsyncBaseTool):
    """
    On-demand data loader tool.

    Loads data with by calling the provided loader function,
    stores in index, and queries for relevant data with a
    natural language query string.

    """

    def __init__(
        self,
        loader: Callable[..., List[Document]],
        index_cls: Type[BaseIndex],
        index_kwargs: Dict,
        metadata: ToolMetadata,
        use_query_str_in_loader: bool = False,
        query_str_kwargs_key: str = "query_str",
    ) -> None:
        """Init params."""
        self._loader = loader
        self._index_cls = index_cls
        self._index_kwargs = index_kwargs
        self._use_query_str_in_loader = use_query_str_in_loader
        self._metadata = metadata
        self._query_str_kwargs_key = query_str_kwargs_key

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @classmethod
    def from_defaults(
        cls,
        reader: BaseReader,
        index_cls: Optional[Type[BaseIndex]] = None,
        index_kwargs: Optional[Dict] = None,
        use_query_str_in_loader: bool = False,
        query_str_kwargs_key: str = "query_str",
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "OnDemandLoaderTool":
        """From defaults."""
        # NOTE: fn_schema should be specified if you want to use as langchain Tool

        index_cls = index_cls or VectorStoreIndex
        index_kwargs = index_kwargs or {}
        if description is None:
            description = f"Tool to load data from {reader.__class__.__name__}"
        if fn_schema is None:
            fn_schema = create_schema_from_function(
                name or "LoadData",
                reader.load_data,
                [(query_str_kwargs_key, str, None)],
            )

        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(
            loader=reader.load_data,
            index_cls=index_cls,
            index_kwargs=index_kwargs,
            use_query_str_in_loader=use_query_str_in_loader,
            query_str_kwargs_key=query_str_kwargs_key,
            metadata=metadata,
        )

    @classmethod
    def from_tool(
        cls,
        tool: FunctionTool,
        index_cls: Optional[Type[BaseIndex]] = None,
        index_kwargs: Optional[Dict] = None,
        use_query_str_in_loader: bool = False,
        query_str_kwargs_key: str = "query_str",
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "OnDemandLoaderTool":
        """From defaults."""
        # NOTE: fn_schema should be specified if you want to use as langchain Tool

        index_cls = index_cls or VectorStoreIndex
        index_kwargs = index_kwargs or {}
        if description is None:
            description = f"Tool to load data from {tool.__class__.__name__}"
        if fn_schema is None:
            fn_schema = create_schema_from_function(
                name or "LoadData", tool._fn, [(query_str_kwargs_key, str, None)]
            )
        metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema,
            return_direct=return_direct,
        )
        return cls(
            loader=tool._fn,
            index_cls=index_cls,
            index_kwargs=index_kwargs,
            use_query_str_in_loader=use_query_str_in_loader,
            query_str_kwargs_key=query_str_kwargs_key,
            metadata=metadata,
        )

    def _parse_args(self, *args: Any, **kwargs: Any) -> Tuple[str, List[Document]]:
        if self._query_str_kwargs_key not in kwargs:
            raise ValueError(
                "Missing query_str in kwargs with parameter name: "
                f"{self._query_str_kwargs_key}"
            )
        if self._use_query_str_in_loader:
            query_str = kwargs[self._query_str_kwargs_key]
        else:
            query_str = kwargs.pop(self._query_str_kwargs_key)

        docs = self._loader(*args, **kwargs)

        return query_str, docs

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        query_str, docs = self._parse_args(*args, **kwargs)

        index = self._index_cls.from_documents(docs, **self._index_kwargs)
        # TODO: add query kwargs
        query_engine = index.as_query_engine()
        response = query_engine.query(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"query": query_str},
            raw_output=response,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Async Call."""
        query_str, docs = self._parse_args(*args, **kwargs)

        index = self._index_cls.from_documents(docs, **self._index_kwargs)
        # TODO: add query kwargs
        query_engine = index.as_query_engine()
        response = await query_engine.aquery(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"query": query_str},
            raw_output=response,
        )
