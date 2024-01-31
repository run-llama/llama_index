from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.embeddings.base import BaseEmbedding


def test_azure_openai_embedding_class():
    names_of_base_classes = [b.__name__ for b in AzureOpenAIEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
