import pytest
import typesense

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.typesense import TypesenseVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TypesenseVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def _store_with_hits(hits):
    client = typesense.Client(
        {
            "api_key": "placeholder-key",
            "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
        }
    )
    client.multi_search.perform = lambda *args, **kwargs: {"results": [{"hits": hits}]}
    return TypesenseVectorStore(client=client)


def _hit(node_id, distance):
    node = TextNode(id_=node_id, text=node_id)
    return {
        "document": {
            "id": node_id,
            "text": node_id,
            "metadata": node_to_metadata_dict(node),
        },
        "vector_distance": distance,
    }


def test_query_returns_similarities_not_distances():
    store = _store_with_hits(
        [_hit("exact", 0.0), _hit("close", 0.05), _hit("far", 1.9)]
    )
    result = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3)
    )

    assert result.ids == ["exact", "close", "far"]
    # Typesense's vector_distance is a cosine distance (0 = identical),
    # but similarities must be higher-is-better.
    assert result.similarities == pytest.approx([1.0, 0.95, -0.9])
