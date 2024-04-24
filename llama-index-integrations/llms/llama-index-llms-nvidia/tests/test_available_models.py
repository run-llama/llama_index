import pytest

from llama_index.llms.nvidia import NVIDIA


@pytest.mark.integration()
def test_available_models() -> None:
    models = NVIDIA().available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
