from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def test_openai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenAIEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
