import builtins
import importlib
import sys
from enum import Enum
from types import ModuleType

import pytest

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


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

    class UpstreamPromptMode(Enum):
        UNSPECIFIED = "unspecified"
        RANK_GPT = "rank_GPT"
        RANK_GPT_APEER = "rank_GPT_APEER"
        LRL = "LRL"
        MONOT5 = "monot5"
        DUOT5 = "duot5"
        LIT5 = "LiT5"

    class Request:
        def __init__(self, query, candidates):
            self.query = query
            self.candidates = candidates

    class Query:
        def __init__(self, text, qid):
            self.text = text
            self.qid = qid

    class Candidate:
        def __init__(self, docid, score, doc):
            self.docid = docid
            self.score = score
            self.doc = doc

    class Reranker:
        coordinator_kwargs = None

        @staticmethod
        def create_model_coordinator(**kwargs):
            Reranker.coordinator_kwargs = kwargs
            return object()

        def __init__(self, model_coordinator):
            self.model_coordinator = model_coordinator

        def rerank(self, request, **kwargs):
            request.candidates.reverse()
            return request

    reranker_module.Reranker = Reranker
    rankllm_module.PromptMode = UpstreamPromptMode
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
    return UpstreamPromptMode, Reranker


def test_import_and_construction_do_not_load_rank_llm(monkeypatch):
    _clear_rankllm_rerank_modules(monkeypatch)
    for module_name in list(sys.modules):
        if module_name == "rank_llm" or module_name.startswith("rank_llm."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def block_rank_llm_import(name, *args, **kwargs):
        if name == "rank_llm" or name.startswith("rank_llm."):
            raise AssertionError("rank_llm must not load before reranking")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_rank_llm_import)
    module = importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    reranker = module.RankLLMRerank(top_n=1, batch_size=1)
    assert issubclass(module.RankLLMRerank, BaseNodePostprocessor)
    assert reranker.prompt_mode.value == "rank_GPT"
    assert module.RankLLMRerank.model_json_schema()["properties"]["prompt_mode"]


def test_prompt_modes_accept_upstream_enums_and_serialize(monkeypatch):
    upstream_prompt_mode, _ = _mock_rankllm_modules(monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)
    module = importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    for prompt_mode in module.base.PromptMode:
        reranker = module.RankLLMRerank(top_n=1, batch_size=1, prompt_mode=prompt_mode)
        assert reranker.model_dump(mode="json")["prompt_mode"] == prompt_mode.value

    reranker = module.RankLLMRerank(
        top_n=1, batch_size=1, prompt_mode=upstream_prompt_mode.DUOT5
    )
    assert reranker.prompt_mode.value == upstream_prompt_mode.DUOT5.value
    with pytest.raises(ValueError):
        module.RankLLMRerank(top_n=1, batch_size=1, prompt_mode="invalid")


def test_reranking_import_error_is_actionable_and_chained(monkeypatch):
    _clear_rankllm_rerank_modules(monkeypatch)
    module = importlib.import_module("llama_index.postprocessor.rankllm_rerank")
    reranker = module.RankLLMRerank(top_n=1, batch_size=1)
    real_import = builtins.__import__

    def raise_vllm_import_error(name, *args, **kwargs):
        if name == "rank_llm.data":
            raise ModuleNotFoundError("No module named 'vllm'", name="vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raise_vllm_import_error)

    with pytest.raises(ImportError, match="could not load RankLLM") as error:
        reranker._postprocess_nodes([], QueryBundle(query_str="query"))
    assert isinstance(error.value.__cause__, ImportError)
    assert error.value.__cause__.name == "vllm"


def test_reranking_converts_mode_and_preserves_scores(monkeypatch):
    upstream_prompt_mode, upstream_reranker = _mock_rankllm_modules(monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)
    module = importlib.import_module("llama_index.postprocessor.rankllm_rerank")
    events = []
    monkeypatch.setattr(
        type(module.base.dispatcher),
        "event",
        lambda _dispatcher, event: events.append(event),
    )
    reranker = module.RankLLMRerank(
        model="monot5",
        top_n=1,
        batch_size=1,
        prompt_mode=module.base.PromptMode.MONOT5,
    )
    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=0.25),
        NodeWithScore(node=TextNode(text="second"), score=0.75),
    ]

    result = reranker._postprocess_nodes(nodes, QueryBundle(query_str="query"))

    assert (
        upstream_reranker.coordinator_kwargs["prompt_mode"]
        is upstream_prompt_mode.MONOT5
    )
    assert [node.node.text for node in result] == ["second"]
    assert [node.score for node in result] == [0.75]
    assert len(events) == 2
