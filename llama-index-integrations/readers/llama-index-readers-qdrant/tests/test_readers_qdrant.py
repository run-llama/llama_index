from llama_index.core.readers.base import BaseReader
from llama_index.readers.qdrant import QdrantReader
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def test_class():
    names_of_base_classes = [b.__name__ for b in QdrantReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_in_memory():
    # qdrant-client 1.16.0 removed `search`; load_data must work on current clients
    reader = QdrantReader(location=":memory:")
    reader._client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    reader._client.upsert(
        collection_name="test_collection",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"doc_id": "a", "text": "first", "metadata": {"k": "x"}},
            ),
            PointStruct(
                id=2,
                vector=[0.3, 0.2, 0.1],
                payload={"doc_id": "b", "text": "second", "metadata": {"k": "y"}},
            ),
        ],
    )

    documents = reader.load_data(
        collection_name="test_collection",
        query_vector=[0.1, 0.2, 0.3],
        should_search_mapping={"text": "first"},
        must_search_mapping={"metadata.k": "x"},
        limit=5,
    )

    assert len(documents) == 1
    assert documents[0].id_ == "a"
    assert documents[0].text == "first"
    assert documents[0].metadata == {"k": "x"}
    assert len(documents[0].embedding) == 3
