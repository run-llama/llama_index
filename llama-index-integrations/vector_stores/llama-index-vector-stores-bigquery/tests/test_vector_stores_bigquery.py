from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.bigquery import BigQueryVectorStore


def test_class():
    """It should inherit from BasePydanticVectorStore"""
    names_of_base_classes = [b.__name__ for b in BigQueryVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
