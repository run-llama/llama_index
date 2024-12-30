import pytest

from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from requests_mock import Mocker


@pytest.fixture(autouse=True)
def mock_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://test_url/v1/models",
        json={
            "data": [
                {"id": "model1"},
            ]
        },
    )


@pytest.mark.integration()
def test_available_models(mode: dict) -> None:
    models = NVIDIARerank(**mode).available_models
    assert models
    assert isinstance(models, list)
    assert all(isinstance(model.id, str) for model in models)
