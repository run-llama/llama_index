from unittest.mock import patch

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from llama_index.vector_stores.firestore import FirestoreVectorStore


@patch("importlib.metadata.version", return_value="0.1.0", autospec=True)
def test_class(_version: str) -> None:
    names_of_base_classes = [b.__name__ for b in FirestoreVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
