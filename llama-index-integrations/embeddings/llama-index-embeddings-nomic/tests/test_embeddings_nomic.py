from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nomic import NomicEmbedding


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in BaseEmbedding.__mro__]
    assert NomicEmbedding.__name__ in names_of_base_classes
