import importlib.machinery
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator, List, NamedTuple, NoReturn, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)

# rank-llm 0.25.7 imports vllm while loading rank_llm.rerank, before any
# backend is selected. Open-source listwise models are the only ones that
# actually call vllm.
_VLLM_REQUIRED_MESSAGE = (
    "vllm is required for RankLLM open-source listwise rerankers "
    "(for example RankZephyr and RankVicuna) and is not installed. "
    "Other backends, such as RankGPT, do not need vllm."
)


class _RankLLMSymbols(NamedTuple):
    prompt_mode: Any
    reranker: Any
    request: Any
    query: Any
    candidate: Any


_symbols: Optional[_RankLLMSymbols] = None


class _UnavailableVllmAttr:
    """Stand-in for vllm attributes so rank-llm can be imported without it."""

    def __init__(self, qualname: str) -> None:
        self._qualname = qualname

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(_VLLM_REQUIRED_MESSAGE)

    def __getattr__(self, name: str) -> "_UnavailableVllmAttr":
        return _UnavailableVllmAttr(f"{self._qualname}.{name}")


class _VllmStubLoader:
    """Load a stand-in vllm package so rank-llm 0.25.7's eager import succeeds."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__path__ = []
        module.__package__ = module.__name__

        def __getattr__(name: str) -> _UnavailableVllmAttr:
            if name.startswith("_"):
                raise AttributeError(name)
            attr = _UnavailableVllmAttr(f"{module.__name__}.{name}")
            setattr(module, name, attr)
            return attr

        module.__getattr__ = __getattr__  # type: ignore[method-assign]


class _VllmStubFinder:
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname != "vllm" and not fullname.startswith("vllm."):
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _VllmStubLoader(),
            is_package=True,
        )


def _iter_exceptions(exc: BaseException) -> Iterator[BaseException]:
    """
    Yield ``exc`` and explicit ``raise ... from`` causes, not ``__context__``.

    ``__context__`` is set when a new exception is raised inside an ``except``
    block. Following it made a later rank-llm failure look like a missing vllm
    install, because the first attempt had already raised ``ModuleNotFoundError``.
    """
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _is_missing_vllm(exc: BaseException) -> bool:
    """
    Return whether ``exc`` was caused by the top-level vllm module being absent.

    A missing ``vllm`` submodule of an installed vllm package is left alone so a
    real vllm install is not replaced with a stand-in.
    """
    for current in _iter_exceptions(exc):
        if isinstance(current, ModuleNotFoundError) and current.name == "vllm":
            return True
    return False


def _is_missing_rank_llm(exc: BaseException) -> bool:
    for current in _iter_exceptions(exc):
        if isinstance(current, ModuleNotFoundError) and (
            current.name == "rank_llm" or (current.name or "").startswith("rank_llm.")
        ):
            return True
    return False


def _drop_partial_rerank_modules() -> None:
    """
    Drop a half-initialized rank_llm.rerank so the import can be retried.

    A failed ``import vllm`` inside ``rank_llm.rerank`` leaves the parent
    package in ``sys.modules``. Python will not run that package's ``__init__``
    again until the entry is removed.
    """
    for name in list(sys.modules):
        if name == "rank_llm.rerank" or name.startswith("rank_llm.rerank."):
            del sys.modules[name]
    rank_llm_pkg = sys.modules.get("rank_llm")
    if rank_llm_pkg is not None and hasattr(rank_llm_pkg, "rerank"):
        delattr(rank_llm_pkg, "rerank")


def _purge_vllm_modules() -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]


@contextmanager
def _temporary_vllm_stub() -> Iterator[None]:
    """Serve import requests for vllm, then remove the stand-ins from sys.modules."""
    finder = _VllmStubFinder()
    _purge_vllm_modules()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        _purge_vllm_modules()


def _import_rankllm_modules() -> _RankLLMSymbols:
    from rank_llm.data import Candidate, Query, Request
    from rank_llm.rerank.rankllm import PromptMode
    from rank_llm.rerank.reranker import Reranker

    return _RankLLMSymbols(
        prompt_mode=PromptMode,
        reranker=Reranker,
        request=Request,
        query=Query,
        candidate=Candidate,
    )


def _raise_rankllm_import_error(exc: ImportError) -> NoReturn:
    if _is_missing_rank_llm(exc):
        raise ImportError(
            "RankLLMRerank requires the rank-llm package. "
            "Install it with `pip install rank-llm`."
        ) from exc
    if _is_missing_vllm(exc):
        raise ImportError(_VLLM_REQUIRED_MESSAGE) from exc
    raise exc


def _load_rankllm_symbols() -> _RankLLMSymbols:
    try:
        return _import_rankllm_modules()
    except ImportError as exc:
        if not _is_missing_vllm(exc):
            _raise_rankllm_import_error(exc)
        # rank-llm 0.25.7 imports vllm from rank_llm.rerank for every backend.
        # Retry with a stand-in so RankGPT and the other non-vllm backends can
        # load. Calling into a vllm model still raises ImportError.
        _drop_partial_rerank_modules()
        try:
            with _temporary_vllm_stub():
                return _import_rankllm_modules()
        except ImportError as retry_exc:
            _raise_rankllm_import_error(retry_exc)


def _rankllm_symbols() -> _RankLLMSymbols:
    global _symbols
    if _symbols is None:
        _symbols = _load_rankllm_symbols()
    return _symbols


class _DeferredPromptMode:
    """PromptMode.RANK_GPT stand-in used when rank-llm cannot be imported yet."""

    name = "RANK_GPT"
    value = "rank_GPT"

    def __repr__(self) -> str:
        return "PromptMode.RANK_GPT"


def _default_prompt_mode() -> Any:
    """
    Return PromptMode.RANK_GPT without importing rank-llm at module import.

    Construction must succeed even when rank-llm 0.25.7 cannot be imported,
    because that import pulls in vllm for every backend. The real enum is
    resolved when reranking actually runs.
    """
    try:
        return _rankllm_symbols().prompt_mode.RANK_GPT
    except ImportError:
        return _DeferredPromptMode()


def _resolve_prompt_mode(prompt_mode: Any, prompt_mode_cls: Any) -> Any:
    if isinstance(prompt_mode, _DeferredPromptMode):
        return prompt_mode_cls.RANK_GPT
    return prompt_mode


class RankLLMRerank(BaseNodePostprocessor):
    """
    RankLLM reranking suite. This class allows access to several reranking models supported by RankLLM. To use a model offered by the RankLLM suite, pass the desired model's hugging face path, found at https://huggingface.co/castorini. e.g., to access LiT5-Distill-base, pass 'castorini/LiT5-Distill-base' as the model name (https://huggingface.co/castorini/LiT5-Distill-base).

    Below are all the rerankers supported with the model name to be passed as an argument to the constructor. Some model have convenience names for ease of use:
        Listwise:
            - OSLLM (Open Source LLM). Takes in a valid Hugging Face model name. e.g., 'Qwen/Qwen2.5-7B-Instruct'
            - RankZephyr. model='rank_zephyr' or 'castorini/rank_zephyr_7b_v1_full'
            - RankVicuna. model='rank_zephyr' or 'castorini/rank_vicuna_7b_v1'
            - RankGPT. Takes in a valid gpt model. e.g., 'gpt-3.5-turbo', 'gpt-4','gpt-3'
            - GenAI. Takes in a valid gemini model. e.g., 'gemini-2.0-flash'
        Pairwise:
            - DuoT5. model='duot5'
        Pointwise:
            - MonoT5. model='monot5'
    """

    model: str = Field(description="Model name.", default="rank_zephyr")
    top_n: Optional[int] = Field(
        description="Number of nodes to return sorted by reranking score."
    )
    window_size: int = Field(
        description="Reranking window size. Applicable only for listwise and pairwise models.",
        default=20,
    )
    batch_size: Optional[int] = Field(
        description="Reranking batch size. Applicable only for pointwise models."
    )
    context_size: int = Field(
        description="Maximum number of tokens for the context window.", default=4096
    )
    prompt_mode: Any = Field(
        description="Prompt format and strategy used when invoking the reranking model.",
        default_factory=_default_prompt_mode,
    )
    num_gpus: int = Field(
        description="Number of GPUs to use for inference if applicable.", default=1
    )
    num_few_shot_examples: int = Field(
        description="Number of few-shot examples to include in the prompt.", default=0
    )
    few_shot_file: Optional[str] = Field(
        description="Path to a file containing few-shot examples, used if few-shot prompting is enabled.",
        default=None,
    )
    use_logits: bool = Field(
        description="Whether to use raw logits for reranking scores instead of probabilities.",
        default=False,
    )
    use_alpha: bool = Field(
        description="Whether to apply an alpha scaling factor in the reranking score calculation.",
        default=False,
    )
    variable_passages: bool = Field(
        description="Whether to allow passages of variable lengths instead of fixed-size chunks.",
        default=False,
    )
    stride: int = Field(
        description="Stride to use when sliding over long documents for reranking.",
        default=10,
    )
    use_azure_openai: bool = Field(
        description="Whether to use Azure OpenAI instead of the standard OpenAI API.",
        default=False,
    )

    _reranker: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "RankLLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        symbols = _rankllm_symbols()
        reranker_cls = symbols.reranker
        request_cls = symbols.request
        query_cls = symbols.query
        candidate_cls = symbols.candidate
        prompt_mode = _resolve_prompt_mode(self.prompt_mode, symbols.prompt_mode)

        kwargs = {
            "model_path": self.model,
            "default_model_coordinator": None,
            "context_size": self.context_size,
            "prompt_mode": prompt_mode,
            "num_gpus": self.num_gpus,
            "use_logits": self.use_logits,
            "use_alpha": self.use_alpha,
            "num_few_shot_examples": self.num_few_shot_examples,
            "few_shot_file": self.few_shot_file,
            "variable_passages": self.variable_passages,
            "interactive": False,
            "window_size": self.window_size,
            "stride": self.stride,
            "use_azure_openai": self.use_azure_openai,
        }
        model_coordinator = reranker_cls.create_model_coordinator(**kwargs)
        self._reranker = reranker_cls(model_coordinator)

        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )

        docs = [
            (node.get_content(metadata_mode=MetadataMode.EMBED), node.get_score())
            for node in nodes
        ]

        request = request_cls(
            query=query_cls(
                text=query_bundle.query_str,
                qid=1,
            ),
            candidates=[
                candidate_cls(
                    docid=index,
                    score=doc[1],
                    doc={
                        "body": doc[0],
                        "headings": "",
                        "title": "",
                        "url": "",
                    },
                )
                for index, doc in enumerate(docs)
            ],
        )

        # scores are maintained the same as generated from the retriever
        permutation = self._reranker.rerank(
            request,
            rank_end=len(request.candidates),
            rank_start=0,
            shuffle_candidates=False,
            logging=False,
            top_k_retrieve=len(request.candidates),
        )

        new_nodes: List[NodeWithScore] = []
        for candidate in permutation.candidates:
            id: int = int(candidate.docid)
            new_nodes.append(NodeWithScore(node=nodes[id].node, score=nodes[id].score))

        if self.top_n is None:
            dispatcher.event(ReRankEndEvent(nodes=new_nodes))
            return new_nodes
        else:
            dispatcher.event(ReRankEndEvent(nodes=new_nodes[: self.top_n]))
            return new_nodes[: self.top_n]
