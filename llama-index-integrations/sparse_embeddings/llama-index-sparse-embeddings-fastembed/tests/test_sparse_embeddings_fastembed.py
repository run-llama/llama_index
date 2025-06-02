import pytest

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.sparse_embeddings.fastembed import FastEmbedSparseEmbedding


def test_class():
    names_of_base_classes = [b.__name__ for b in FastEmbedSparseEmbedding.__mro__]
    assert BaseSparseEmbedding.__name__ in names_of_base_classes


def test_e2e():
    embed_model = FastEmbedSparseEmbedding(model_name="Qdrant/bm25")
    texts = ["hello", "world"]
    embeddings = embed_model.get_text_embedding_batch(texts)
    assert len(embeddings) == len(texts)

    queries = ["foo"]
    embedding = embed_model.get_query_embedding(queries[0])
    assert len(embedding) == 1


@pytest.mark.asyncio
async def test_e2e_async():
    embed_model = FastEmbedSparseEmbedding(model_name="Qdrant/bm25")
    texts = ["hello", "world"]
    embeddings = await embed_model.aget_text_embedding_batch(texts)
    assert len(embeddings) == len(texts)

    queries = ["foo"]
    embedding = await embed_model.aget_query_embedding(queries[0])
    assert len(embedding) == 1
