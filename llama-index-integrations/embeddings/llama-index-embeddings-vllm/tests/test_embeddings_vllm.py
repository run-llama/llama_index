import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.vllm import VllmEmbedding


def test_vllmembedding_class():
    names_of_base_classes = [b.__name__ for b in VllmEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_embedding_retry():
    try:
        embed_model = VllmEmbedding()
    except Exception:
        # will fail in certain environments
        # skip test if it fails
        pytest.skip("Skipping test due to environment issue")
        return

    # Test successful embedding
    result = embed_model._embed(["This is a test sentence"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])
