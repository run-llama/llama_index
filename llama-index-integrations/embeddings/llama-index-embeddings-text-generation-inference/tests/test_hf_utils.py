from typing import Any, Dict

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.text_generation_inference import (
    TextGenerationInferenceEmbedding,
)
from pytest import MonkeyPatch


def mock_tgi_embedding(self: Any, *args: Any, **kwargs: Dict[str, Any]) -> Any:
    """Mock TextGenerationInferenceEmbedding."""
    super(TextGenerationInferenceEmbedding, self).__init__(
        model_name="fake",
        tokenizer_name="fake",
        model="fake",
        tokenizer="fake",
    )
    return


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.embeddings.huggingface.TextGenerationInferenceEmbedding.__init__",
        mock_tgi_embedding,
    )

    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)

    # Test str
    embed_model = resolve_embed_model("local")
    assert isinstance(embed_model, TextGenerationInferenceEmbedding)

    # Test LCEmbeddings
    embed_model = resolve_embed_model(TextGenerationInferenceEmbedding())
    assert isinstance(embed_model, TextGenerationInferenceEmbedding)
