"""
Get evaluation utils.

NOTE: These are beta functions, might change.

"""

import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from llama_index.core.async_utils import asyncio_module, asyncio_run
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.evaluation.base import EvaluationResult

if TYPE_CHECKING:
    from llama_index.core.llama_dataset import LabelledRagDataset


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
    """
    Get responses.

    Sync version of aget_responses.

    """
    return asyncio_run(aget_responses(*args, **kwargs))


def get_results_df(
    eval_results_list: List[Dict[str, List[EvaluationResult]]],
    names: List[str],
    metric_keys: List[str],
) -> Any:
    """
    Get results df.

    Args:
        eval_results_list (List[Dict[str, List[EvaluationResult]]]):
            List of evaluation results.
        names (List[str]):
            Names of the evaluation results.
        metric_keys (List[str]):
            List of metric keys to get.

    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Pandas is required to get results dataframes. Please install it with `pip install pandas`."
        )

    metric_dict = defaultdict(list)
    metric_dict["names"] = names
    for metric_key in metric_keys:
        for eval_results in eval_results_list:
            mean_score = np.array(
                [r.score or 0.0 for r in eval_results[metric_key]]
            ).mean()
            metric_dict[metric_key].append(mean_score)
    return pd.DataFrame(metric_dict)


def _download_llama_dataset_from_hub(llama_dataset_id: str) -> "LabelledRagDataset":
    """Uses a subprocess and llamaindex-cli to download a dataset from llama-hub."""
    from llama_index.core.llama_dataset import LabelledRagDataset

    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                [
                    "llamaindex-cli",
                    "download-llamadataset",
                    f"{llama_dataset_id}",
                    "--download-dir",
                    f"{tmp}",
                ]
            )
            return LabelledRagDataset.from_json(f"{tmp}/rag_dataset.json")  # type: ignore
        except FileNotFoundError as err:
            raise ValueError(
                "No dataset associated with the supplied `llama_dataset_id`"
            ) from err


def default_parser(eval_response: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Default parser function for evaluation response.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.

    """
    if not eval_response.strip():
        # Return None or default values if the response is empty
        return None, "No response"

    score_str, reasoning_str = eval_response.split("\n", 1)

    try:
        score = float(score_str)
    except ValueError:
        score = None

    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning
