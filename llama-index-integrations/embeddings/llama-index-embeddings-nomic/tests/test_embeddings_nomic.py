from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.nomic import NomicEmbedding


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in NomicEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes
