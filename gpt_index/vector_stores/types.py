"""Vector store index types."""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from enum import Enum
from gpt_index.data_structs.node_v2 import Node
from gpt_index.docstore import BaseDocumentStore


@dataclass
class NodeEmbeddingResult:
    """Node embedding result.

    Args:
        id (str): Node id
        node (Node): Node
        embedding (List[float]): Embedding
        doc_id (str): Document id

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


class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class VectorStoreQueryConfig:
    """Vector store query config."""

    similarity_top_k: int = 1
    doc_ids: Optional[List[str]] = None
    # NOTE: current mode
    mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT
    # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    alpha: Optional[float] = None
    # NOTE: only for time-weighted search atm
    use_time_decay: bool = False
    time_decay_rate: float = 0.99
    last_accessed_key: Optional[str] = "__last_accessed__"
    # whether to refresh last_accessed_key
    time_access_refresh: bool = True


@dataclass
class VectorStoreQuery:
    """Vector store query."""

    # dense embedding
    query_embedding: Optional[List[float]] = None
    # similarity_top_k: int = 1
    query_str: Optional[str] = None

    # used for embedding-only indexes; can look up node in docstore.
    docstore: Optional[BaseDocumentStore] = None

    query_config: VectorStoreQueryConfig = field(default_factory=VectorStoreQueryConfig)

    @property
    def similarity_top_k(self) -> int:
        """Get similarity top k."""
        return self.query_config.similarity_top_k

    @property
    def doc_ids(self) -> Optional[List[str]]:
        """Get doc ids."""
        return self.query_config.doc_ids

    @property
    def mode(self) -> VectorStoreQueryMode:
        """Get mode."""
        return self.query_config.mode

    @property
    def alpha(self) -> Optional[float]:
        """Get alpha."""
        return self.query_config.alpha

    @property
    def use_time_decay(self) -> bool:
        """Get use time decay."""
        return self.query_config.use_time_decay

    @property
    def time_decay_rate(self) -> float:
        """Get time decay rate."""
        return self.query_config.time_decay_rate

    # # NOTE: current mode
    # mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT

    # # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    # alpha: Optional[float] = None

    # # NOTE: only for time-weighted search atm
    # use_time_decay: bool = False
    # time_decay_rate: float = 0.99


@runtime_checkable
class VectorStore(Protocol):
    """Abstract vector store protocol."""

    stores_text: bool
    is_embedding_query: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        ...

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
        self,
        query: VectorStoreQuery,
        # query_embedding: List[float],
        # similarity_top_k: int,
        # doc_ids: Optional[List[str]] = None,
        # query_str: Optional[str] = None,
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        ...
