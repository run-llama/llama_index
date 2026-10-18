from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llama_index.llms.ovhcloud import OVHcloud
from llama_index.llms.ovhcloud.utils import Model, get_available_models_dynamic


def _client(*ids: str) -> MagicMock:
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=model_id) for model_id in ids]
    )
    return client


def test_get_available_models_dynamic() -> None:
    models = get_available_models_dynamic(_client("model-a", "model-b"))
    assert models == [Model(id="model-a"), Model(id="model-b")]


def test_available_models_uses_the_models_endpoint() -> None:
    llm = OVHcloud(model="model-a", api_key="fake-api-key")
    with patch.object(OVHcloud, "_get_client", return_value=_client("a", "b", "c")):
        assert [m.id for m in llm.available_models] == ["a", "b", "c"]


def test_available_models_falls_back_to_current_model() -> None:
    llm = OVHcloud(model="model-a", api_key="fake-api-key")
    with patch.object(OVHcloud, "_get_client", side_effect=Exception("boom")):
        assert [m.id for m in llm.available_models] == ["model-a"]
