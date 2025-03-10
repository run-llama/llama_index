from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


def test_embedding_class():
    emb = GoogleGenAIEmbedding()
    assert isinstance(emb, BaseEmbedding)
