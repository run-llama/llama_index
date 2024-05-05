import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding


@pytest.mark.integration()
def test_available_models(mode: dict) -> None:
    models = NVIDIAEmbedding().mode(**mode).available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
