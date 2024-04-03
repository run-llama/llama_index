from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_itrex import ItrexQuantizedBgeEmbedding


def test_itrex_embedding_class():
    names_of_base_classes = [b.__name__ for b in ItrexQuantizedBgeEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
