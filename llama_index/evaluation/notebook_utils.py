"""Notebook utils."""

from collections import defaultdict
from typing import List, Optional

import pandas as pd

from llama_index.evaluation.retrieval.base import RetrievalEvalResult

DEFAULT_METRIC_KEYS = ["hit_rate", "mrr"]


def get_retrieval_results_df(
    names: List[str],
    results_arr: List[List[RetrievalEvalResult]],
    metric_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Display retrieval results."""
    metric_keys = metric_keys or DEFAULT_METRIC_KEYS

    avg_metrics_dict = defaultdict(list)
    for name, eval_results in zip(names, results_arr):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)
        results_df = pd.DataFrame(metric_dicts)

        for metric_key in metric_keys:
            if metric_key not in results_df.columns:
                raise ValueError(f"Metric key {metric_key} not in results_df")
            avg_metrics_dict[metric_key].append(results_df[metric_key].mean())

    return pd.DataFrame({"retrievers": names, **avg_metrics_dict})
