"""Get evaluation utils.

NOTE: These are beta functions, might change.

"""

import asyncio
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from llama_index.legacy.async_utils import asyncio_module
from llama_index.legacy.core.base_query_engine import BaseQueryEngine
from llama_index.legacy.evaluation.base import EvaluationResult


async def aget_responses(
    questions: List[str], query_engine: BaseQueryEngine, show_progress: bool = False
) -> List[str]:
    """Get responses."""
    tasks = []
    for question in questions:
        tasks.append(query_engine.aquery(question))
    asyncio_mod = asyncio_module(show_progress=show_progress)
    return await asyncio_mod.gather(*tasks)


def get_responses(
    *args: Any,
    **kwargs: Any,
) -> List[str]:
    """Get responses.

    Sync version of aget_responses.

    """
    return asyncio.run(aget_responses(*args, **kwargs))


def get_results_df(
    eval_results_list: List[EvaluationResult], names: List[str], metric_keys: List[str]
) -> pd.DataFrame:
    """Get results df.

    Args:
        eval_results_list (List[EvaluationResult]):
            List of evaluation results.
        names (List[str]):
            Names of the evaluation results.
        metric_keys (List[str]):
            List of metric keys to get.

    """
    metric_dict = defaultdict(list)
    metric_dict["names"] = names
    for metric_key in metric_keys:
        for eval_results in eval_results_list:
            mean_score = np.array([r.score for r in eval_results[metric_key]]).mean()
            metric_dict[metric_key].append(mean_score)
    return pd.DataFrame(metric_dict)


def default_parser(eval_response: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Default parser function for evaluation response.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    score_str, reasoning_str = eval_response.split("\n", 1)
    score = float(score_str)
    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning
