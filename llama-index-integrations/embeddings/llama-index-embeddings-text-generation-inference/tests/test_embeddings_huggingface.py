from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.text_generation_inference import (
    TextGenerationInferenceEmbedding,
)


def test_textgenerationinferenceembedding_class():
    names_of_base_classes = [
        b.__name__ for b in TextGenerationInferenceEmbedding.__mro__
    ]
    assert BaseEmbedding.__name__ in names_of_base_classes
