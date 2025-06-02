from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_api import (
    HuggingFaceInferenceAPIEmbedding,
)


def test_huggingfaceapiembedding_class():
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceInferenceAPIEmbedding.__mro__
    ]
    assert BaseEmbedding.__name__ in names_of_base_classes
