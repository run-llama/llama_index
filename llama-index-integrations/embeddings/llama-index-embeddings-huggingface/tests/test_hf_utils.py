from typing import Any, Dict

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)
from pytest import MonkeyPatch


def mock_hf_embeddings(self: Any, *args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock HuggingFaceEmbeddings."""
    super(HuggingFaceEmbedding, self).__init__(
        model_name="fake",
        tokenizer_name="fake",
        model="fake",
        tokenizer="fake",
    )
    return


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding.__init__",
        mock_hf_embeddings,
    )

    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)

    # Test str
    embed_model = resolve_embed_model("local")
    assert isinstance(embed_model, HuggingFaceEmbedding)

    # Test LCEmbeddings
    embed_model = resolve_embed_model(HuggingFaceEmbedding())
    assert isinstance(embed_model, HuggingFaceEmbedding)


def test_get_pooling_mode_sets_request_timeout(monkeypatch: MonkeyPatch) -> None:
    import requests
    from llama_index.embeddings.huggingface.utils import get_pooling_mode

    calls: list = []

    class _Response:
        def json(self) -> Dict[str, Any]:
            return {"pooling_mode_mean_tokens": True}

    def fake_get(url: str, **kwargs: Any) -> _Response:
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    assert get_pooling_mode("some/model") == "mean"
    assert calls[0].get("timeout")
