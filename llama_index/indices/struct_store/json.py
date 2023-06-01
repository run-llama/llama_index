from typing import Any, Optional, Sequence, Union, List, Dict

from llama_index.data_structs.node import Node
from llama_index.data_structs.table import JSONStructDatapoint
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.struct_store.base import BaseGPTStructStoreIndex

JSONType = Union[Dict[str, "JSONType"], List["JSONType"], str, int, float, bool, None]


class GPTJSONIndex(BaseGPTStructStoreIndex[JSONStructDatapoint]):
    index_struct_cls = JSONStructDatapoint

    def __init__(
        self,
        json_value: JSONType,
        json_schema: JSONType,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[JSONStructDatapoint] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""

        if nodes is not None:
            raise ValueError("We currently do not support indexing documents or nodes.")
        self.json_value = json_value
        self.json_schema = json_schema

        # index_struct = index_struct or self.index_struct_cls(fields=self.json_value)
        super().__init__(
            nodes=[],
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError("Not supported")

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.indices.struct_store.json_query import GPTNLJSONQueryEngine

        return GPTNLJSONQueryEngine(self, json_schema=self.json_schema, **kwargs)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> JSONStructDatapoint:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        raise NotImplementedError("We currently do not support inserting documents.")
