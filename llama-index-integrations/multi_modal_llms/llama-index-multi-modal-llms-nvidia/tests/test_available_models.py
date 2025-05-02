import pytest

from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal


@pytest.mark.integration()
def test_available_models() -> None:
    models = NVIDIAMultiModal().available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
