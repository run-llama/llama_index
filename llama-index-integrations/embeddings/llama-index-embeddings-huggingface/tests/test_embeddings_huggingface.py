from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
)
import pytest


def test_huggingfaceembedding_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_huggingfaceapiembedding_class():
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceInferenceAPIEmbedding.__mro__
    ]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_input_validation():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    # Test empty input
    with pytest.raises(ValueError, match="Input text cannot be empty or whitespace"):
        embed_model._validate_input("")

    # Test whitespace input
    with pytest.raises(ValueError, match="Input text cannot be empty or whitespace"):
        embed_model._validate_input("   ")

    # Test valid input
    embed_model._validate_input("This is a valid input")  # Should not raise


def test_embedding_retry():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    # Test successful embedding
    result = embed_model._embed(["This is a test sentence"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])
