from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.nomic_multimodal import NomicMultiModalEmbedding


def test_multimodal_embeddings_nomic():
    names_of_base_classes = [b.__name__ for b in NomicMultiModalEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes
