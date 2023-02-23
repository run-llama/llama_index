"""Vector store index types."""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Type, TypeVar

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


VS = TypeVar("VS", bound="VectorStore")

# TODO: decide whether best interface is protocol or abstract class
class VectorStore(ABC):
    """Abstract vector store."""

    stores_text: bool

    @abstractmethod
    @property
    def client(self) -> Any:
        """Get client."""

    @abstractmethod
    @property
    def config_dict(self) -> dict:
        """Get config dict."""

    @abstractmethod
    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding rentsults to vector store."""

    @abstractmethod
    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""

    @abstractmethod
    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Query vector store."""

    @classmethod
    def load_from_dict(cls: Type[VS], config_dict: dict, **kwargs) -> VS:
        """Load vector store from existing dictionary.

        The user may need to specify additional kwargs in order to
        initialize the vector store. If a kwarg is specified which
        overrides a value in the config_dict, the value in the
        kwargs will be used.
        
        """
        config_dict = config_dict.copy()
        config_dict.update(kwargs)
        return cls(**config_dict)
        