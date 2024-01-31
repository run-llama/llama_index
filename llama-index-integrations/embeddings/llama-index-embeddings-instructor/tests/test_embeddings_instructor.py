from llama_index.core.embeddings.base import BaseEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding


def test_embedding_class():
    emb = InstructorEmbedding()
    assert isinstance(emb, BaseEmbedding)
