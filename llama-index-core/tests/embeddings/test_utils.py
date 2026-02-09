from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.utils import load_embedding, resolve_embed_model
from pytest import MonkeyPatch
import pytest
from pathlib import Path


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)


def test_load_embedding_empty_file(tmp_path: Path) -> None:
    """Test load_embedding with an empty file to cover the newly added error check."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()

    with pytest.raises(ValueError, match="is empty"):
        load_embedding(str(empty_file))
