from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.astra_db.base import AstraDBVectorStore


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in AstraDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
