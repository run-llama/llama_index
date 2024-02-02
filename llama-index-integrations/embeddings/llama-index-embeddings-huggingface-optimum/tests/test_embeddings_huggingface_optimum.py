from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding


def test_optimumembedding_class():
    names_of_base_classes = [b.__name__ for b in OptimumEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
