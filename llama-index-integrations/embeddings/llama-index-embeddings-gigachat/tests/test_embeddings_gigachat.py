from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gigachat import GigaChatEmbedding


def test_embedding_function():
    names_of_base_classes = [b.__name__ for b in GigaChatEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
