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


def test_model_kwargs_omit_default_best_of() -> None:
    from llama_index.llms.vllm import VllmServer

    remote = VllmServer(api_url="http://localhost:8000")

    assert "best_of" not in remote._model_kwargs


def test_model_kwargs_include_explicit_best_of() -> None:
    from llama_index.llms.vllm import VllmServer

    remote = VllmServer(api_url="http://localhost:8000", best_of=2)

    assert remote._model_kwargs["best_of"] == 2
