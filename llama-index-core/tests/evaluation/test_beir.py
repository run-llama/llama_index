import inspect

from llama_index.core.evaluation.benchmarks.beir import BeirEvaluator


def test_download_datasets_uses_none_default() -> None:
    """
    `_download_datasets(datasets=...)` must default to `None`, not a
    shared `["nfcorpus"]` list (B006 - mutable default argument).
    """
    sig = inspect.signature(BeirEvaluator._download_datasets)
    assert sig.parameters["datasets"].default is None


def test_run_uses_none_defaults_for_list_arguments() -> None:
    """
    `run(datasets=..., metrics_k_values=...)` must default to `None`,
    not shared `["nfcorpus"]` / `[3, 10]` lists (B006).
    """
    sig = inspect.signature(BeirEvaluator.run)
    assert sig.parameters["datasets"].default is None
    assert sig.parameters["metrics_k_values"].default is None
