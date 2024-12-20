from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.adapter import AdapterEmbeddingModel


def test_class():
    names_of_base_classes = [b.__name__ for b in AdapterEmbeddingModel.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
