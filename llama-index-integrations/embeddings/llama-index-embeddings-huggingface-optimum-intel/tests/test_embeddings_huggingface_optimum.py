from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_optimum_intel import IntelEmbedding


def test_optimum_intel_embedding_class():
    names_of_base_classes = [b.__name__ for b in IntelEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
