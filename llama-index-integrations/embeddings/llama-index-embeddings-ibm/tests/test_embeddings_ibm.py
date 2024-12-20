from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ibm import WatsonxEmbeddings


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in WatsonxEmbeddings.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
