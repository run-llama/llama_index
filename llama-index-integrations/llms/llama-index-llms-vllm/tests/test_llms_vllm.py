from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.callbacks import CallbackManager


def test_embedding_class():
    from llama_index.llms.vllm import Vllm

    names_of_base_classes = [b.__name__ for b in Vllm.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_server_class():
    from llama_index.llms.vllm import VllmServer

    names_of_base_classes = [b.__name__ for b in VllmServer.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_server_callback() -> None:
    from llama_index.llms.vllm import VllmServer

    callback_manager = CallbackManager()
    remote = VllmServer(
        api_url="http://localhost:8000",
        model="modelstub",
        max_new_tokens=123,
        callback_manager=callback_manager,
    )
    assert remote.callback_manager == callback_manager
    del remote


@patch("llama_index.llms.vllm.base.get_response", return_value=["ok"])
@patch("llama_index.llms.vllm.base.post_http_request")
def test_server_complete_includes_model(mock_post: MagicMock, mock_get: MagicMock):
    from llama_index.llms.vllm import VllmServer

    mock_post.return_value = MagicMock()
    server = VllmServer(
        api_url="http://localhost:8000/v1/completions",
        model="test-model",
    )

    server.complete("Hello world")

    assert mock_post.call_count == 1
    payload = mock_post.call_args[0][1]
    assert payload["model"] == "test-model"
    assert payload["prompt"] == "Hello world"
    mock_get.assert_called_once_with(mock_post.return_value)


@patch("llama_index.llms.vllm.base.get_response", return_value=["ok"])
@patch("llama_index.llms.vllm.base.post_http_request")
def test_server_complete_respects_custom_model(
    mock_post: MagicMock, mock_get: MagicMock
):
    from llama_index.llms.vllm import VllmServer

    mock_post.return_value = MagicMock()
    server = VllmServer(
        api_url="http://localhost:8000/v1/completions",
        model="default-model",
    )

    server.complete("Hello world", model="override")

    payload = mock_post.call_args[0][1]
    assert payload["model"] == "override"
