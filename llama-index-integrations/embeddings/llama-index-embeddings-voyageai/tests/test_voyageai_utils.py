from pytest import MonkeyPatch
from typing import Any, Dict

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.voyageai import VoyageEmbedding


def mock_voyageai_embeddings(self: Any, *args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock OpenAIEmbedding."""
    super(VoyageEmbedding, self).__init__(voyage_api_key="fake")
    return


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.embeddings.voyageai.VoyageEmbedding.__init__",
        mock_voyageai_embeddings,
    )

    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)

    # Test BaseEmbedding
    embed_model = resolve_embed_model(VoyageEmbedding())
    assert isinstance(embed_model, VoyageEmbedding)
