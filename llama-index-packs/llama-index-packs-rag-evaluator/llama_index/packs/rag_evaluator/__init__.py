import warnings

warnings.warn(
    "llama-index-packs-rag-evaluator is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.rag_evaluator.base import RagEvaluatorPack

__all__ = ["RagEvaluatorPack"]
