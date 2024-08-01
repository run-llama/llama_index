import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.google import (
    GeminiEmbedding,
    GooglePaLMEmbedding,
    GoogleUnivSentEncoderEmbedding,
)

try:
    import tensorflow_hub
except ImportError:
    tensorflow_hub = None  # type: ignore


@pytest.mark.skipif(tensorflow_hub is None, reason="tensorflow_hub not installed")
def test_tf_embedding_class():
    emb = GoogleUnivSentEncoderEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_embedding_class():
    emb = GeminiEmbedding()
    assert isinstance(emb, BaseEmbedding)

    emb = GooglePaLMEmbedding()
    assert isinstance(emb, BaseEmbedding)
