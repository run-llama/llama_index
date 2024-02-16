from typing import Any, Dict

# pants: no-infer-dep
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)  # pants: no-infer-dep
from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep
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


def mock_openai_embeddings(self: Any, *args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock OpenAIEmbedding."""
    super(OpenAIEmbedding, self).__init__(
        api_key="fake", api_base="fake", api_version="fake"
    )
    return


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding.__init__",
        mock_hf_embeddings,
    )
    monkeypatch.setattr(
        "llama_index.embeddings.openai.OpenAIEmbedding.__init__",
        mock_openai_embeddings,
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

    # Test BaseEmbedding
    embed_model = resolve_embed_model(OpenAIEmbedding())
    assert isinstance(embed_model, OpenAIEmbedding)
