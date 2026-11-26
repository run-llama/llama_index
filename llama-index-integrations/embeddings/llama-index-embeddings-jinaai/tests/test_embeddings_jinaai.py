import asyncio

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding


def test_embedding_class():
    emb = JinaEmbedding()
    assert isinstance(emb, BaseEmbedding)
    assert isinstance(emb, MultiModalEmbedding)


@pytest.mark.parametrize("image_source", ["url", "local"])
def test_aget_image_embedding(mocker, tmp_path, image_source):
    if image_source == "local":
        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"image bytes")
        image = str(image_path)
        expected_input = [{"bytes": "aW1hZ2UgYnl0ZXM="}]
    else:
        image = "https://example.com/image.png"
        expected_input = [{"url": image}]

    post = mocker.patch("aiohttp.ClientSession.post")
    response = post.return_value.__aenter__.return_value
    response.raise_for_status = mocker.Mock()
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
    emb = JinaEmbedding(api_key="test-key")

    result = asyncio.run(emb.aget_image_embedding(image))

    assert result == [0.1, 0.2, 0.3]
    assert post.call_args.kwargs["json"]["input"] == expected_input
    response.json.assert_awaited_once()
    post.return_value.__aexit__.assert_awaited_once()


def test_get_image_embedding(mocker):
    post = mocker.patch("requests.Session.post")
    post.return_value.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    }
    emb = JinaEmbedding(api_key="test-key")

    result = emb.get_image_embedding("https://example.com/image.png")

    assert result == [0.1, 0.2, 0.3]


def test_aget_image_embedding_batch(mocker):
    post = mocker.patch("aiohttp.ClientSession.post")
    response = post.return_value.__aenter__.return_value
    response.raise_for_status = mocker.Mock()
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ]
    }
    emb = JinaEmbedding(api_key="test-key")
    images = ["https://example.com/first.png", "https://example.com/second.png"]

    result = asyncio.run(emb.aget_image_embedding_batch(images))

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert post.call_args.kwargs["json"]["input"] == [
        {"url": images[0]},
        {"url": images[1]},
    ]


def test_aget_image_embedding_propagates_errors(mocker):
    post = mocker.patch("aiohttp.ClientSession.post")
    response = post.return_value.__aenter__.return_value
    response.json.return_value = {"detail": "provider failed"}
    response.raise_for_status = mocker.Mock(side_effect=RuntimeError("provider failed"))
    emb = JinaEmbedding(api_key="test-key")

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(emb.aget_image_embedding("https://example.com/image.png"))

    post.return_value.__aexit__.assert_awaited_once()
