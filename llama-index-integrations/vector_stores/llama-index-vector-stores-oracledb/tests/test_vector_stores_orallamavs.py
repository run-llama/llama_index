from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.oracledb import OraLlamaVS


def test_class():
    names_of_base_classes = [b.__name__ for b in OraLlamaVS.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
