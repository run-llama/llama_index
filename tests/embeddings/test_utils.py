from typing import Any, Dict
from pytest import MonkeyPatch

from llama_index.bridge.langchain import HuggingFaceBgeEmbeddings
from llama_index.embeddings import OpenAIEmbedding, LangchainEmbedding
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.token_counter.mock_embed_model import MockEmbedding


def mock_hf_embeddings(*args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock HuggingFaceEmbeddings."""
    return


def mock_openai_embeddings(*args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock OpenAIEmbedding."""
    return


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.bridge.langchain.HuggingFaceBgeEmbeddings.__init__",
        mock_hf_embeddings,
    )
    monkeypatch.setattr(
        "llama_index.embeddings.OpenAIEmbedding.__init__", mock_openai_embeddings
    )

    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)

    # Test str
    embed_model = resolve_embed_model("local")
    assert isinstance(embed_model, LangchainEmbedding)

    # Test LCEmbeddings
    embed_model = resolve_embed_model(HuggingFaceBgeEmbeddings())
    assert isinstance(embed_model, LangchainEmbedding)

    # Test BaseEmbedding
    embed_model = resolve_embed_model(OpenAIEmbedding())
    assert isinstance(embed_model, OpenAIEmbedding)
