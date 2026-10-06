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


def test_best_of_excluded_when_none() -> None:
    """best_of=None (the default) must not be forwarded to SamplingParams.

    vLLM 0.26+ removed best_of from SamplingParams; passing it unconditionally
    as None causes TypeError on initialization even when the caller never set it.
    """
    from llama_index.llms.vllm import Vllm

    llm = Vllm.__new__(Vllm)
    llm.__dict__.update(
        {
            "model": "stub",
            "temperature": 1.0,
            "max_new_tokens": 512,
            "n": 1,
            "best_of": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "ignore_eos": False,
            "stop": None,
            "logprobs": None,
            "top_k": -1,
            "top_p": 1.0,
        }
    )
    assert "best_of" not in llm._model_kwargs


def test_best_of_included_when_set() -> None:
    """best_of is forwarded when explicitly provided."""
    from llama_index.llms.vllm import Vllm

    llm = Vllm.__new__(Vllm)
    llm.__dict__.update(
        {
            "model": "stub",
            "temperature": 1.0,
            "max_new_tokens": 512,
            "n": 1,
            "best_of": 4,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "ignore_eos": False,
            "stop": None,
            "logprobs": None,
            "top_k": -1,
            "top_p": 1.0,
        }
    )
    assert llm._model_kwargs.get("best_of") == 4
