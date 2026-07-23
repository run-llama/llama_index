from unittest.mock import MagicMock

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.callbacks import CallbackManager


def test_model_kwargs_best_of_conditional():
    """Omits best_of when None, includes it when set. Regression for vLLM>=0.19 (#21371)."""
    from llama_index.llms.vllm import Vllm

    prop = Vllm._model_kwargs.fget

    obj = MagicMock(
        temperature=0.1,
        max_new_tokens=64,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        ignore_eos=False,
        stop=[],
        logprobs=None,
        top_k=-1,
        top_p=1.0,
        best_of=None,
    )
    assert "best_of" not in prop(obj)

    obj.best_of = 3
    assert prop(obj)["best_of"] == 3


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
