from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.objectbox import ObjectBoxVectorStore


def test_class():
    """Ensures that BasePydanticVectorStore is one of the parent classes of ObjectBoxVectorStore."""
    names_of_base_classes = [b.__name__ for b in ObjectBoxVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
