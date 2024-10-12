import asyncio
import os
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding


def test_embedding_class():
    emb = ZhipuAIEmbedding(model="", api_key="")
    assert isinstance(emb, BaseEmbedding)


@pytest.mark.skipif(
    os.environ.get("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
def test_get_text_embedding():
    emb = ZhipuAIEmbedding(
        model="embedding-2",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
    )
    response = emb.get_general_text_embedding("who are you?")
    assert len(response) == 1024
    response = asyncio.run(emb.aget_general_text_embedding("who are you?"))
    assert len(response) == 1024


@pytest.mark.skipif(
    os.environ.get("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
def test_dimensions_setting():
    emb = ZhipuAIEmbedding(
        model="embedding-3",
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        dimensions=256,
    )
    response = emb.get_general_text_embedding("who are you?")
    assert len(response) == 256
