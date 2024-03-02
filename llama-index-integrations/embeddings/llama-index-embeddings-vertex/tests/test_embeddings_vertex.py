import unittest
from unittest.mock import patch

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding

from llama_index.embeddings.vertex.base import (
    VertexTextEmbedding,
    VertexMultiModalEmbedding,
)


def test_embedding_class():
    emb = VertexTextEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_multimodal_embedding_class():
    emb = VertexMultiModalEmbedding()
    assert isinstance(emb, MultiModalEmbedding)


class TestVertexAIInit(unittest.TestCase):
    @patch("vertexai.init")
    def test_init(self, mock_init):
        mock_init.return_value = None


if __name__ == "__main__":
    unittest.main()
