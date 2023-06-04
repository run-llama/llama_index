"""Ad-hoc data loader tool.

Tool that wraps any data loader, and is able to load data on-demand.

"""


from llama_index.tools.types import BaseTool, ToolMetadata
from llama_index.readers.base import BaseReader
from typing import Any, List, Optional, Dict, Type, Tuple, Callable
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.vector_store import GPTVectorStoreIndex
from pydantic import BaseModel, create_model
from inspect import signature


def create_schema_from_function(
    name: str,
    func: Callable[..., Any],
    additional_fields: Optional[List[Tuple[str, Type, Any]]] = None,
) -> Type[BaseModel]:
    """Create schema from function."""
    # NOTE: adapted from langchain.tools.base
    fields = {}
    params = signature(func).parameters
    for param_name in params.keys():
        param_type = params[param_name].annotation
        param_default = params[param_name].default
        if param_default is params[param_name].empty:
            param_default = None
        fields[param_name] = (param_type, param_default)

    additional_fields = additional_fields or []
    for field_name, field_type, field_default in additional_fields:
        fields[field_name] = (field_type, field_default)

    return create_model(name, **fields)  # type: ignore


class AdhocLoaderTool(BaseTool):
    """Ad-hoc data loader tool.

    Loads data with query interface, stores in index, and queries
    for relevant data with a natural language query string.

    """

    def __init__(
        self,
        reader: BaseReader,
        index_cls: Type[BaseGPTIndex],
        index_kwargs: Dict,
        metadata: ToolMetadata,
        use_query_str_in_loader: bool = False,
        query_str_kwargs_key: str = "query_str",
    ) -> None:
        """Init params."""
        self._reader = reader
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
        index_cls: Optional[Type[BaseGPTIndex]] = None,
        index_kwargs: Optional[Dict] = None,
        use_query_str_in_loader: bool = False,
        query_str_kwargs_key: str = "query_str",
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "AdhocLoaderTool":
        """From defaults."""
        # NOTE: fn_schema should be specified if you want to use as langchain Tool

        index_cls = index_cls or GPTVectorStoreIndex
        index_kwargs = index_kwargs or {}
        if description is None:
            description = f"Tool to load data from {reader.__class__.__name__}"
        if fn_schema is None:
            fn_schema = create_schema_from_function(
                "LoadData", reader.load_data, [("query_str", str, None)]
            )

        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(
            reader=reader,
            index_cls=index_cls,
            index_kwargs=index_kwargs,
            use_query_str_in_loader=use_query_str_in_loader,
            query_str_kwargs_key=query_str_kwargs_key,
            metadata=metadata,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call."""
        if self._query_str_kwargs_key not in kwargs:
            raise ValueError(
                "Missing query_str in kwargs with parameter name: "
                f"{self._query_str_kwargs_key}"
            )
        if self._use_query_str_in_loader:
            query_str = kwargs[self._query_str_kwargs_key]
        else:
            query_str = kwargs.pop(self._query_str_kwargs_key)
        docs = self._reader.load_data(*args, **kwargs)
        index = self._index_cls.from_documents(docs, **self._index_kwargs)
        # TODO: add query kwargs
        query_engine = index.as_query_engine()
        response = query_engine.query(query_str)
        return str(response)
