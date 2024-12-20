from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding


def test_class():
    names_of_base_classes = [b.__name__ for b in FastEmbedEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
