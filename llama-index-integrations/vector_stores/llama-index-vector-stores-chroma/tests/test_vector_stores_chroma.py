from dataclasses import dataclass
from typing import Any, Dict, Optional

from llama_index.core.schema import ImageNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.chroma import ChromaVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in ChromaVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@dataclass
class _FakeChromaCollection:
    query_payload: Optional[Dict[str, Any]] = None
    get_payload: Optional[Dict[str, Any]] = None

    def query(self, **kwargs: Any) -> Dict[str, Any]:
        if self.query_payload is None:
            raise AssertionError("query_payload not provided")
        return self.query_payload

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        if self.get_payload is None:
            raise AssertionError("get_payload not provided")
        return self.get_payload


def _build_metadata(node: ImageNode) -> Dict[str, Any]:
    metadata = node_to_metadata_dict(node, remove_text=True, flat_metadata=True)
    # Mimic ChromaVectorStore.add: replace None values with empty strings
    for key, value in list(metadata.items()):
        if value is None:
            metadata[key] = ""
    return metadata


def test_query_rehydrates_image_nodes_with_missing_documents() -> None:
    node = ImageNode(
        text="",
        image_url="http://example.com/image.png",
        id_="image-node-id",
        embedding=[0.1, 0.2, 0.3],
    )
    metadata = _build_metadata(node)
    fake_collection = _FakeChromaCollection(
        query_payload={
            "ids": [[node.node_id]],
            "documents": [[None]],
            "metadatas": [[metadata]],
            "distances": [[0.0]],
        }
    )
    store = ChromaVectorStore(chroma_collection=fake_collection)

    result = store._query(
        query_embeddings=[0.1, 0.2, 0.3],
        n_results=1,
        where={},
    )

    assert result.nodes is not None
    assert isinstance(result.nodes[0], ImageNode)
    assert result.nodes[0].image_url == node.image_url


def test_get_handles_missing_document_text() -> None:
    node = ImageNode(
        text="",
        image_url="http://example.com/image.png",
        id_="image-node-id",
        embedding=[0.1, 0.2, 0.3],
    )
    metadata = _build_metadata(node)
    fake_collection = _FakeChromaCollection(
        get_payload={
            "ids": [node.node_id],
            "documents": [None],
            "metadatas": [metadata],
        }
    )
    store = ChromaVectorStore(chroma_collection=fake_collection)

    result = store._get(limit=1, where={})

    assert result.nodes is not None
    assert isinstance(result.nodes[0], ImageNode)
    assert result.nodes[0].image_url == node.image_url
