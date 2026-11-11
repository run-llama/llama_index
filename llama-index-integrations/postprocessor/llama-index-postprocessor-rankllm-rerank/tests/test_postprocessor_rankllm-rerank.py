import importlib
import sys
from enum import Enum
from types import ModuleType

import pytest

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle

RANKLLM_RERANK_MODULES = [
    "llama_index.postprocessor.rankllm_rerank",
    "llama_index.postprocessor.rankllm_rerank.base",
]


def _clear_rankllm_rerank_modules(monkeypatch):
    for module_name in RANKLLM_RERANK_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def _mock_rankllm_modules(monkeypatch):
    rank_llm_module = ModuleType("rank_llm")
    rank_llm_module.__path__ = []

    rerank_module = ModuleType("rank_llm.rerank")
    rerank_module.__path__ = []

    reranker_module = ModuleType("rank_llm.rerank.reranker")
    rankllm_module = ModuleType("rank_llm.rerank.rankllm")
    data_module = ModuleType("rank_llm.data")

    class Reranker:
        pass

    class PromptMode(Enum):
        RANK_GPT = "rank_gpt"

    class Request:
        pass

    class Query:
        pass

    class Candidate:
        pass

    reranker_module.Reranker = Reranker
    rankllm_module.PromptMode = PromptMode
    data_module.Request = Request
    data_module.Query = Query
    data_module.Candidate = Candidate

    rank_llm_module.rerank = rerank_module
    rank_llm_module.data = data_module
    rerank_module.reranker = reranker_module
    rerank_module.rankllm = rankllm_module

    monkeypatch.setitem(sys.modules, "rank_llm", rank_llm_module)
    monkeypatch.setitem(sys.modules, "rank_llm.rerank", rerank_module)
    monkeypatch.setitem(sys.modules, "rank_llm.rerank.reranker", reranker_module)
    monkeypatch.setitem(sys.modules, "rank_llm.rerank.rankllm", rankllm_module)
    monkeypatch.setitem(sys.modules, "rank_llm.data", data_module)

    return PromptMode, reranker_module


def test_class(monkeypatch):
    # rank_llm is mocked here: importing the real rank_llm.rerank package pulls
    # in transformers/vllm model code (e.g. rank_llm's lit5 T5 modeling), which
    # breaks at import time with transformers >= 4.56 until rank-llm catches up
    # upstream. The unit test only verifies our class hierarchy.
    _mock_rankllm_modules(monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    names_of_base_classes = [b.__name__ for b in rankllm_rerank.RankLLMRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_import_with_prompt_mode_from_rankllm_module(monkeypatch):
    prompt_mode, reranker_module = _mock_rankllm_modules(monkeypatch)
    assert not hasattr(reranker_module, "PromptMode")

    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    names_of_base_classes = [b.__name__ for b in rankllm_rerank.RankLLMRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes
    reranker = rankllm_rerank.RankLLMRerank(top_n=1, batch_size=1)
    assert reranker.prompt_mode == prompt_mode.RANK_GPT


class _BrokenModule(ModuleType):
    """
    A module whose attributes raise ImportError when accessed.

    Simulates importing rank_llm.rerank when an optional backend
    dependency such as vllm is missing: the import chain fails with
    ModuleNotFoundError, a subclass of ImportError.
    """

    def __getattr__(self, name):
        raise ImportError("No module named 'vllm'")


def _mock_broken_rankllm_modules(monkeypatch):
    _mock_rankllm_modules(monkeypatch)

    broken_rankllm_module = _BrokenModule("rank_llm.rerank.rankllm")
    broken_reranker_module = _BrokenModule("rank_llm.rerank.reranker")
    monkeypatch.setitem(sys.modules, "rank_llm.rerank.rankllm", broken_rankllm_module)
    monkeypatch.setitem(sys.modules, "rank_llm.rerank.reranker", broken_reranker_module)


def test_import_when_rankllm_rerank_import_fails(monkeypatch):
    # Regression test for #23339: RankLLMRerank must be importable even
    # when importing rank_llm.rerank fails, for example because vllm is
    # not installed for a backend that does not use vLLM.
    _mock_broken_rankllm_modules(monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    names_of_base_classes = [b.__name__ for b in rankllm_rerank.RankLLMRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes

    reranker = rankllm_rerank.RankLLMRerank(top_n=1, batch_size=1)
    assert reranker.prompt_mode.name == "RANK_GPT"

    # The missing dependency is only reported when reranking is
    # actually attempted, and with a clear error.
    with pytest.raises(ImportError, match="could not import rank_llm"):
        reranker._postprocess_nodes([], QueryBundle(query_str="test"))
