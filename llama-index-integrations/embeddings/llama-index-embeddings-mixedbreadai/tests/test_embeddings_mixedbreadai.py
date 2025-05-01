import os

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.mixedbreadai import (
    MixedbreadAIEmbedding,
    EncodingFormat,
    TruncationStrategy,
)


def test_embedding_class():
    emb = MixedbreadAIEmbedding(api_key="token")
    assert isinstance(emb, BaseEmbedding)


@pytest.mark.skipif(
    os.environ.get("MXBAI_API_KEY") is None, reason="Mixedbread AI API key required"
)
def test_sync_embedding():
    emb = MixedbreadAIEmbedding(
        api_key=os.environ["MXBAI_API_KEY"],
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        encoding_format=EncodingFormat.INT_8,
        truncation_strategy=TruncationStrategy.START,
    )

    emb.get_query_embedding("Who is german and likes bread?")


@pytest.mark.skipif(
    os.environ.get("MXBAI_API_KEY") is None, reason="Mixedbread AI API key required"
)
@pytest.mark.asyncio
async def test_async_embedding():
    emb = MixedbreadAIEmbedding(
        api_key=os.environ["MXBAI_API_KEY"],
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        encoding_format=EncodingFormat.FLOAT,
        truncation_strategy=TruncationStrategy.START,
    )

    await emb.aget_query_embedding("Who is german and likes bread?")
