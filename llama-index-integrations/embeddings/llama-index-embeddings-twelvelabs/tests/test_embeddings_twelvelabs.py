from unittest.mock import MagicMock

import pytest
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.twelvelabs import TwelveLabsEmbedding
from llama_index.embeddings.twelvelabs.base import _first_vector


def test_class():
    names_of_base_classes = [b.__name__ for b in TwelveLabsEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TwelveLabsEmbedding()


def _vec_response(key="text_embedding", dim=8):
    response = MagicMock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {
        "model_name": "marengo3.0",
        key: {"segments": [{"float": [0.1] * dim}]},
    }
    return response


def _embedder():
    embed = TwelveLabsEmbedding(api_key="tlk_test")
    embed._session = MagicMock()
    return embed


def test_text_embedding():
    embed = _embedder()
    embed._session.post.return_value = _vec_response("text_embedding")
    vec = embed.get_text_embedding("a cat playing piano")
    assert len(vec) == 8
    # text path posts to /embed with multipart fields (no file).
    _, kwargs = embed._session.post.call_args
    assert "files" in kwargs
    assert kwargs["files"]["text"][1] == "a cat playing piano"


def test_query_embedding():
    embed = _embedder()
    embed._session.post.return_value = _vec_response("text_embedding")
    assert len(embed.get_query_embedding("query")) == 8


def test_image_embedding_url():
    embed = _embedder()
    embed._session.post.return_value = _vec_response("image_embedding")
    vec = embed.get_image_embedding("https://example.com/cat.jpg")
    assert len(vec) == 8
    _, kwargs = embed._session.post.call_args
    assert kwargs["files"]["image_url"][1] == "https://example.com/cat.jpg"


def test_embed_http_error_raises():
    embed = _embedder()
    bad = MagicMock()
    bad.ok = False
    bad.status_code = 400
    bad.text = "parameter_invalid"
    embed._session.post.return_value = bad
    with pytest.raises(RuntimeError):
        embed.get_text_embedding("x")


@pytest.mark.parametrize(
    "payload",
    [
        {"text_embedding": {"segments": [{"float": [1.0, 2.0]}]}},
        {"image_embedding": {"segments": [{"float": [1.0, 2.0]}]}},
    ],
)
def test_first_vector(payload):
    assert _first_vector(payload) == [1.0, 2.0]


def test_first_vector_missing_raises():
    with pytest.raises(RuntimeError):
        _first_vector({"unexpected": True})
