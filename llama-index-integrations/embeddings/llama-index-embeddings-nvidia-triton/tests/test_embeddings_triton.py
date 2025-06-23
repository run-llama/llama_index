from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nvidia_triton import NvidiaTritonEmbedding


def test_embedding_class():
    emb = NvidiaTritonEmbedding(
        model_name="", client_kwargs={"ssl": False}
    )
    assert isinstance(emb, BaseEmbedding)
