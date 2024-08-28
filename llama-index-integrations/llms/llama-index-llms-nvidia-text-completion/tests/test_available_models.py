import pytest

from llama_index.llms.nvidia_text_completion import NVIDIATextCompletion


@pytest.mark.integration()
def test_available_models() -> None:
    models = NVIDIATextCompletion().available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
