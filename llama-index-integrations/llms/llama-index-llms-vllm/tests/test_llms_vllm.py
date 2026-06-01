import sys
import types
import warnings

import pytest
from unittest.mock import MagicMock

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.callbacks import CallbackManager


# ---------------------------------------------------------------------------
# vLLM stubs
# ---------------------------------------------------------------------------


def _make_vllm_stub(*, accept_best_of: bool) -> types.ModuleType:
    """Return a minimal vllm module stub.

    When *accept_best_of* is False the SamplingParams signature omits
    ``best_of``, reproducing the behaviour of vLLM >= 0.19.0.
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
                self.kwargs = {k: v for k, v in locals().items() if k != "self"}

    class FakeLLM:
        def __init__(self, model=None, tensor_parallel_size=1,
                     trust_remote_code=False, dtype="auto",
                     download_dir=None, **kw) -> None:
            pass

        def generate(self, prompts, sampling_params):
            out = MagicMock()
            out.outputs[0].text = "hello"
            return [out]

    vllm_mod.SamplingParams = FakeSamplingParams
    vllm_mod.LLM = FakeLLM
    return vllm_mod


@pytest.fixture()
def vllm_new(monkeypatch):
    """Stub for vLLM >= 0.19.0: SamplingParams has no ``best_of``."""
    stub = _make_vllm_stub(accept_best_of=False)
    monkeypatch.setitem(sys.modules, "vllm", stub)
    return stub


@pytest.fixture()
def vllm_old(monkeypatch):
    """Stub for vLLM < 0.19.0: SamplingParams accepts ``best_of``."""
    stub = _make_vllm_stub(accept_best_of=True)
    monkeypatch.setitem(sys.modules, "vllm", stub)
    return stub


# ---------------------------------------------------------------------------
# SamplingParams compatibility tests
# ---------------------------------------------------------------------------


def test_complete_default_best_of_no_error(vllm_new):
    """best_of=None (default) must not be passed to SamplingParams."""
    from llama_index.llms.vllm import Vllm

    llm = Vllm(model="stub")
    result = llm.complete("Hello")
    assert isinstance(result, CompletionResponse)
    assert result.text == "hello"


def test_complete_explicit_best_of_warns_and_succeeds(vllm_new):
    """Explicitly set best_of should warn and be silently dropped on new vLLM."""
    from llama_index.llms.vllm import Vllm

    llm = Vllm(model="stub", best_of=3)
    with pytest.warns(UserWarning, match="best_of"):
        result = llm.complete("Hello")
    assert isinstance(result, CompletionResponse)
    assert result.text == "hello"


def test_complete_best_of_forwarded_on_old_vllm(vllm_old):
    """best_of is forwarded without a warning on vLLM versions that support it."""
    from llama_index.llms.vllm import Vllm

    llm = Vllm(model="stub", best_of=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = llm.complete("Hello")

    assert isinstance(result, CompletionResponse)
    assert result.text == "hello"
    assert not any("best_of" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------


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
