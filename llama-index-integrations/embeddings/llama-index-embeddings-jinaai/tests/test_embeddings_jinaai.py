import asyncio
import base64
from unittest.mock import AsyncMock

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding


def test_embedding_class():
    emb = JinaEmbedding()
    assert isinstance(emb, BaseEmbedding)
    assert isinstance(emb, MultiModalEmbedding)


@pytest.mark.parametrize("local_image", [False, True])
def test_aget_image_embedding(local_image, tmp_path, monkeypatch):
    if local_image:
        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"image data")
        monkeypatch.chdir(tmp_path)
        image = image_path.name
        expected_input = [{"bytes": base64.b64encode(b"image data").decode()}]
    else:
        image = "https://example.com/image.png"
        expected_input = [{"url": image}]

    emb = JinaEmbedding(api_key="test-key")
    emb._api.aget_embeddings = AsyncMock(return_value=[[0.1, 0.2]])

    assert asyncio.run(emb.aget_image_embedding(image)) == [0.1, 0.2]
    emb._api.aget_embeddings.assert_awaited_once_with(input=expected_input)
