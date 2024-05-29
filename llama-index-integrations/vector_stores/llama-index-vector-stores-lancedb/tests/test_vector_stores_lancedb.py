import tantivy  # noqa
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex


def test_class():
    names_of_base_classes = [b.__name__ for b in LanceDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_vector_query(index: VectorStoreIndex) -> None:
    retriever = index.as_retriever()
    response = retriever.retrieve("test1")
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


def test_fts_query(index: VectorStoreIndex) -> None:
    try:
        response = index.as_retriever(
            vector_store_kwargs={"query_type": "fts"}
        ).retrieve("test")
    except Warning as e:
        pass

    response = index.as_retriever(vector_store_kwargs={"query_type": "fts"}).retrieve(
        "test1"
    )
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


def test_hybrid_query(index: VectorStoreIndex) -> None:
    response = index.as_retriever(
        vector_store_kwargs={"query_type": "hybrid"}
    ).retrieve("test")

    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


def test_delete(index: VectorStoreIndex) -> None:
    index.delete(doc_id="test-0")
    assert index.vector_store._table.count_rows() == 2
