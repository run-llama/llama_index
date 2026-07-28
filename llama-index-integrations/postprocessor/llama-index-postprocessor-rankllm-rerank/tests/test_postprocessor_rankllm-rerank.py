import importlib
import sys
from enum import Enum
from types import ModuleType

from llama_index.core.postprocessor.types import BaseNodePostprocessor


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


def test_class():
    from llama_index.postprocessor.rankllm_rerank import RankLLMRerank

    names_of_base_classes = [b.__name__ for b in RankLLMRerank.__mro__]
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
