from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.indices.managed.postgresml import PostgresMLIndex
from llama_index.indices.managed.postgresml import PostgresMLRetriever


def test_class():
    names_of_base_classes = [b.__name__ for b in PostgresMLIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PostgresMLRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes
