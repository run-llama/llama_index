from http import HTTPStatus
from unittest.mock import MagicMock, patch

from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.dashscope.base import get_multimodal_embedding


def test_dashscope_embedding_class():
    names_of_base_classes = [b.__name__ for b in DashScopeEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes


def test_get_multimodal_embedding_parses_new_list_response_shape():
    """
    Regression: newer DashScope API versions return a list of embedding
    objects under "embeddings" instead of a single "embedding" key. Without
    handling this shape, the real embedding vector was never extracted.
    """
    mock_response = MagicMock()
    mock_response.status_code = HTTPStatus.OK
    mock_response.output = {
        "embeddings": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "type": "text"}
        ]
    }

    with patch("dashscope.MultiModalEmbedding.call", return_value=mock_response):
        result = get_multimodal_embedding(
            model="multimodal-embedding-one-peace-v1",
            input=[{"factor": 1, "text": "hello"}],
        )

    assert result == [0.1, 0.2, 0.3]


def test_get_multimodal_embedding_parses_legacy_single_response_shape():
    """
    The older DashScope response shape (a single "embedding" key) must
    keep working for accounts/regions still on the previous API version.
    """
    mock_response = MagicMock()
    mock_response.status_code = HTTPStatus.OK
    mock_response.output = {"embedding": [0.4, 0.5, 0.6]}

    with patch("dashscope.MultiModalEmbedding.call", return_value=mock_response):
        result = get_multimodal_embedding(
            model="multimodal-embedding-one-peace-v1",
            input=[{"factor": 1, "text": "hello"}],
        )

    assert result == [0.4, 0.5, 0.6]


def test_get_multimodal_embedding_returns_empty_list_on_failure():
    mock_response = MagicMock()
    mock_response.status_code = HTTPStatus.BAD_REQUEST

    with patch("dashscope.MultiModalEmbedding.call", return_value=mock_response):
        result = get_multimodal_embedding(
            model="multimodal-embedding-one-peace-v1",
            input=[{"factor": 1, "text": "hello"}],
        )

    assert result == []
