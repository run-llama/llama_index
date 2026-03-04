from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.indices.managed.vertexai import VertexAIIndex
from llama_index.indices.managed.vertexai import VertexAIRetriever


def test_class():
    names_of_base_classes = [b.__name__ for b in VertexAIIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in VertexAIRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes
