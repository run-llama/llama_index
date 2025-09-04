from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.autoembeddings import ChonkieAutoEmbedding


def test_class_init() -> None:
    emb = ChonkieAutoEmbedding(model_name="all-MiniLM-L6-v2")
    assert isinstance(emb, BaseEmbedding)
