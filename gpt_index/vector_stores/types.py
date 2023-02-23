"""Vector store index types."""


from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

from gpt_index.data_structs.data_structs import Node


@dataclass
class NodeEmbeddingResult:
    """Node embedding result.

    Args:
        id (str): Node id
        node (Node): Node
        embedding (List[float]): Embedding

    """

    id: str
    node: Node
    embedding: List[float]
    doc_id: str


@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""

    nodes: Optional[List[Node]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None


class VectorStore(Protocol):
    """Abstract vector store protocol."""

    stores_text: bool

    # TODO: this is more suitable as a class attribute (but that doesn't
    # play nice with protocols)
    # @property
    # def stores_text(self) -> bool:
    #     """Whether the vector store stores text."""
    #     ...

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    @property
    def config_dict(self) -> dict:
        """Get config dict."""
        ...

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to vector store."""
        ...

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        ...

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        ...
