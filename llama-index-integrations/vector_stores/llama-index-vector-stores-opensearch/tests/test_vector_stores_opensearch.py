from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.opensearch import OpensearchVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in OpensearchVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
