import sys
import types
import warnings
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import CompletionResponse
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


# ---------------------------------------------------------------------------
# vLLM SamplingParams compatibility (#21371)
# ---------------------------------------------------------------------------


def _make_vllm_stub(*, accept_best_of: bool) -> types.ModuleType:
    """Minimal ``vllm`` module stub.

    When *accept_best_of* is False the SamplingParams signature omits
    ``best_of``, matching vLLM >= 0.19.0.
    """
    vllm_mod = types.ModuleType("vllm")

    if accept_best_of:

        class FakeSamplingParams:
            def __init__(
                self,
                temperature: float = 1.0,
                max_tokens: int = 512,
                n: int = 1,
                top_p: float = 1.0,
                top_k: int = -1,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                ignore_eos: bool = False,
                stop=None,
                logprobs=None,
                best_of=None,
            ) -> None:
                self.kwargs = {k: v for k, v in locals().items() if k != "self"}

    else:

        class FakeSamplingParams:  # type: ignore[no-redef]
            def __init__(
                self,
                temperature: float = 1.0,
                max_tokens: int = 512,
                n: int = 1,
                top_p: float = 1.0,
                top_k: int = -1,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                ignore_eos: bool = False,
                stop=None,
                logprobs=None,
            ) -> None:
                # Mirror real Python: unexpected kwargs raise TypeError.
                # (This __init__ has no **kwargs, so Python enforces it.)
                self.kwargs = {k: v for k, v in locals().items() if k != "self"}

    class FakeLLM:
        def __init__(
            self,
            model=None,
            tensor_parallel_size=1,
            trust_remote_code=False,
            dtype="auto",
            download_dir=None,
            **kw,
        ) -> None:
            pass

        def generate(self, prompts, sampling_params):
            out = MagicMock()
            out.outputs = [MagicMock(text="ok")]
            return [out]

    vllm_mod.SamplingParams = FakeSamplingParams
    vllm_mod.LLM = FakeLLM
    return vllm_mod


def _vllm_construct(**overrides):
    """Pydantic construct without loading a real vLLM model."""
    from llama_index.llms.vllm import Vllm

    fields = dict(
        model="stub-model",
        temperature=0.0,
        max_new_tokens=16,
        n=1,
        best_of=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_p=1.0,
        top_k=-1,
        stop=None,
        ignore_eos=False,
        logprobs=None,
        is_chat_model=False,
    )
    fields.update(overrides)
    return Vllm.model_construct(**fields)


def _make_vllm_instance(monkeypatch, *, accept_best_of: bool, **field_overrides):
    """Build a Vllm instance without loading a real model."""
    stub = _make_vllm_stub(accept_best_of=accept_best_of)
    monkeypatch.setitem(sys.modules, "vllm", stub)

    llm = _vllm_construct(**field_overrides)
    llm._client = stub.LLM()
    return llm, stub


def test_model_kwargs_omits_unset_best_of():
    """Default best_of=None must not appear in kwargs passed to SamplingParams."""
    llm = _vllm_construct(temperature=0.1, max_new_tokens=32)

    assert "best_of" not in llm._model_kwargs
    assert llm._model_kwargs["temperature"] == 0.1


def test_model_kwargs_includes_set_best_of():
    llm = _vllm_construct(temperature=0.1, max_new_tokens=32, best_of=2)

    assert llm._model_kwargs["best_of"] == 2


def test_complete_works_when_vllm_removed_best_of(monkeypatch):
    """#21371: vLLM >= 0.19 SamplingParams has no best_of — complete() must work."""
    llm, stub = _make_vllm_instance(monkeypatch, accept_best_of=False)
    # Even if a caller still sets best_of on the wrapper, filter it out.
    llm.best_of = 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = llm.complete("hello")

    assert isinstance(result, CompletionResponse)
    assert result.text == "ok"
    assert any("best_of" in str(w.message) for w in caught)


def test_complete_passes_best_of_when_supported(monkeypatch):
    llm, stub = _make_vllm_instance(monkeypatch, accept_best_of=True, best_of=3)

    captured = {}

    def capture_generate(prompts, sampling_params):
        captured["params"] = sampling_params
        out = MagicMock()
        out.outputs = [MagicMock(text="ok")]
        return [out]

    llm._client.generate = capture_generate
    result = llm.complete("hello")
    assert result.text == "ok"
    assert captured["params"].kwargs.get("best_of") == 3


def test_filter_sampling_params_drops_unknown_keys(monkeypatch):
    llm, _stub = _make_vllm_instance(monkeypatch, accept_best_of=False)
    filtered = llm._filter_sampling_params(
        {"temperature": 0.2, "best_of": 2, "not_a_param": 1},
        valid_keys={"temperature", "max_tokens"},
    )
    assert filtered == {"temperature": 0.2}
