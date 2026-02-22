from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_langchain import HuggingFaceLangChainEmbedding


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceLangChainEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
