from typing import Any, Dict, List, Optional

from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)


class MockVectorStore(VectorStore):
    stores_text: bool = True

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        self._config_dict = config_dict or {
            "attr1": 0,
            "attr2": "attr2_val",
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockVectorStore":
        return cls(config_dict)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self._config_dict

    @property
    def client(self) -> Any:
        """Get client."""
        return None

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to vector store."""
        raise NotImplementedError()

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        raise NotImplementedError()

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query vector store."""
        raise NotImplementedError()
