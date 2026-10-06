import importlib
import sys
from enum import Enum
from types import ModuleType

import pytest

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


class _BlockVllmImport:
    """Reject every vllm import so the test does not depend on a local install."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ModuleNotFoundError(
                f"No module named {fullname!r}",
                name=fullname,
            )


def _install_fake_rankllm(tmp_path, monkeypatch):
    """Write a rank-llm 0.25.7-shaped package whose rerank import pulls in vllm."""
    files = {
        "rank_llm/__init__.py": "",
        "rank_llm/data.py": """
class Query:
    def __init__(self, text, qid):
        self.text = text
        self.qid = qid


class Candidate:
    def __init__(self, docid, score, doc):
        self.docid = docid
        self.score = score
        self.doc = doc


class Request:
    def __init__(self, query, candidates):
        self.query = query
        self.candidates = candidates
""",
        "rank_llm/rerank/__init__.py": "from .reranker import Reranker\n",
        "rank_llm/rerank/rankllm.py": """
from enum import Enum


class PromptMode(Enum):
    RANK_GPT = "rank_GPT"
""",
        "rank_llm/rerank/listwise/__init__.py": (
            "from .rank_listwise_os_llm import RankListwiseOSLLM\n"
        ),
        "rank_llm/rerank/listwise/rank_listwise_os_llm.py": """
import vllm
from vllm.outputs import RequestOutput

RankListwiseOSLLM = RequestOutput
""",
        "rank_llm/rerank/reranker.py": """
import vllm
from vllm.outputs import RequestOutput

from rank_llm.rerank.listwise import RankListwiseOSLLM


class _Permutation:
    def __init__(self, candidates):
        self.candidates = candidates


class Reranker:
    def __init__(self, model_coordinator):
        self._model_coordinator = model_coordinator

    @staticmethod
    def create_model_coordinator(**kwargs):
        # Touch the eagerly imported names so a failed vllm import cannot be
        # papered over by skipping this module.
        assert RankListwiseOSLLM is RequestOutput
        model_path = kwargs["model_path"]
        if "gpt" not in model_path:
            vllm.LLM(model=model_path)
        return model_path

    def rerank(self, request, **kwargs):
        return _Permutation(list(reversed(request.candidates)))
""",
    }
    for relative_path, source in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    for name in list(sys.modules):
        if name == "rank_llm" or name.startswith("rank_llm.") or name == "vllm":
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", [_BlockVllmImport(), *sys.meta_path])


def test_import_does_not_load_rank_llm(monkeypatch):
    for name in list(sys.modules):
        if name == "rank_llm" or name.startswith("rank_llm."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    _clear_rankllm_rerank_modules(monkeypatch)

    importlib.import_module("llama_index.postprocessor.rankllm_rerank")

    assert "rank_llm" not in sys.modules
    assert "rank_llm.rerank" not in sys.modules


def test_non_vllm_backend_works_when_rankllm_imports_vllm(tmp_path, monkeypatch):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    _install_fake_rankllm(tmp_path, monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")
    assert "rank_llm" not in sys.modules

    reranker = rankllm_rerank.RankLLMRerank(
        model="gpt-4o-mini",
        top_n=2,
        batch_size=1,
    )
    assert reranker.prompt_mode.name == "RANK_GPT"
    assert "vllm" not in sys.modules

    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=0.1),
        NodeWithScore(node=TextNode(text="second"), score=0.2),
    ]
    reranked = reranker.postprocess_nodes(nodes, QueryBundle(query_str="query"))

    assert [node.node.get_content() for node in reranked] == ["second", "first"]
    assert "vllm" not in sys.modules


def test_constructor_survives_rankllm_import_failure(tmp_path, monkeypatch):
    package = tmp_path / "rank_llm" / "rerank"
    package.mkdir(parents=True)
    (tmp_path / "rank_llm" / "__init__.py").write_text("")
    (package / "__init__.py").write_text(
        "raise ImportError('rank_llm.rerank failed during import')\n"
    )
    (tmp_path / "rank_llm" / "data.py").write_text("")

    for name in list(sys.modules):
        if name == "rank_llm" or name.startswith("rank_llm."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")
    reranker = rankllm_rerank.RankLLMRerank(top_n=1, batch_size=1)

    assert reranker.prompt_mode.name == "RANK_GPT"
    assert "vllm" not in sys.modules


def test_vllm_backend_reports_missing_vllm(tmp_path, monkeypatch):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    _install_fake_rankllm(tmp_path, monkeypatch)
    _clear_rankllm_rerank_modules(monkeypatch)

    rankllm_rerank = importlib.import_module("llama_index.postprocessor.rankllm_rerank")
    reranker = rankllm_rerank.RankLLMRerank(model="rank_zephyr", top_n=1, batch_size=1)
    nodes = [NodeWithScore(node=TextNode(text="first"), score=0.1)]

    with pytest.raises(ImportError, match="do not need vllm"):
        reranker.postprocess_nodes(nodes, QueryBundle(query_str="query"))
