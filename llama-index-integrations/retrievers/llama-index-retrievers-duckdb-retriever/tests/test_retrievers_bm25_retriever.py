from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.duckdb_retriever.base import (
    DuckDBRetriever,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in DuckDBRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes
