import pytest

from llama_index.postprocessor.nvidia_rerank import NVIDIARerank


@pytest.mark.integration()
def test_available_models(mode: dict) -> None:
    models = NVIDIARerank().mode(**mode).available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
