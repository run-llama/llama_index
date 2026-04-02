import warnings

warnings.warn(
    "llama-index-packs-evaluator-benchmarker is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.evaluator_benchmarker.base import EvaluatorBenchmarkerPack

__all__ = ["EvaluatorBenchmarkerPack"]
