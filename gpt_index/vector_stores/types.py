"""Vector store index types."""


from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

from gpt_index.data_structs.data_structs import Node


@dataclass
class NodeEmbeddingResult:
    id: str
    node: Node
    embedding: List[float]
    doc_id: str


@dataclass
class VectorStoreQueryResult:
    nodes: Optional[List[Node]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[int]] = None


class VectorStore(Protocol):
    @property
    def client(self) -> Any:
        ...

    @property
    def config_dict(self) -> dict:
        ...

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        ...

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        ...

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        ...
