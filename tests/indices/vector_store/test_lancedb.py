"""Test LanceDB index."""
from typing import Any, List, Optional, Dict

from unittest.mock import patch

from gpt_index.indices.vector_store import GPTLanceDBIndex
from gpt_index.vector_stores.types import NodeEmbeddingResult
from tests.mock_utils.mock_decorator import patch_common


class MockLanceDBVectorStore:
    stores_text: bool = True

    def __init__(
        self,
        uri: str,
        table_name: str = "vectors",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.connection = None
        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockLanceDBVectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "uri": self.uri,
            "table_name": self.table_name,
            "nprobes": self.nprobes,
            "refine_factor": self.refine_factor,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        return []


@patch_common
@patch(
    "gpt_index.indices.vector_store.vector_indices.LanceDBVectorStore",
    MockLanceDBVectorStore,
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE",
    {MockLanceDBVectorStore: "mock_type"},
)
@patch(
    "gpt_index.vector_stores.registry.VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS",
    {"mock_type": MockLanceDBVectorStore},
)
def test_save_load(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test we can save and load."""
    index = GPTLanceDBIndex.from_documents(documents=[], uri=".")
    save_dict = index.save_to_dict()
    loaded_index = GPTLanceDBIndex.load_from_dict(
        save_dict,
    )
    assert isinstance(loaded_index, GPTLanceDBIndex)
