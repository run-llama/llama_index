from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding


def test_langchain_embedding_class():
    names_of_base_classes = [b.__name__ for b in LangchainEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
