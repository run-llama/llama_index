from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
)
from llama_index.core.embeddings.base import BaseEmbedding


def test_huggingfaceembedding_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_huggingfaceapiembedding_class():
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceInferenceAPIEmbedding.__mro__
    ]
    assert BaseEmbedding.__name__ in names_of_base_classes
