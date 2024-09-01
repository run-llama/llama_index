"""Get evaluation utils.

NOTE: These are beta functions, might change.

"""

import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from llama_index.core.async_utils import asyncio_module, asyncio_run
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.ingestion.api_utils import get_client

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
    """Get responses.

    Sync version of aget_responses.

    """
    return asyncio_run(aget_responses(*args, **kwargs))


def get_results_df(
    eval_results_list: List[EvaluationResult], names: List[str], metric_keys: List[str]
) -> Any:
    """Get results df.

    Args:
        eval_results_list (List[EvaluationResult]):
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
            mean_score = np.array([r.score for r in eval_results[metric_key]]).mean()
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
            return LabelledRagDataset.from_json(f"{tmp}/rag_dataset.json")
        except FileNotFoundError as err:
            raise ValueError(
                "No dataset associated with the supplied `llama_dataset_id`"
            ) from err


def upload_eval_dataset(
    dataset_name: str,
    questions: Optional[List[str]] = None,
    llama_dataset_id: Optional[str] = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    overwrite: bool = False,
    append: bool = False,
) -> str:
    """Upload questions to platform dataset."""
    from llama_cloud import ProjectCreate
    from llama_cloud.types.eval_question_create import EvalQuestionCreate

    if questions is None and llama_dataset_id is None:
        raise ValueError(
            "Must supply either a list of `questions`, or a `llama_dataset_id` to import from llama-hub."
        )

    client = get_client(base_url=base_url, api_key=api_key)

    project = client.projects.upsert_project(request=ProjectCreate(name=project_name))
    assert project.id is not None

    existing_datasets = client.projects.get_datasets_for_project(project_id=project.id)

    # check if dataset already exists
    cur_dataset = None
    for dataset in existing_datasets:
        if dataset.name == dataset_name:
            if overwrite:
                assert dataset.id is not None
                client.evals.delete_dataset(dataset_id=dataset.id)
                break
            elif not append:
                raise ValueError(
                    f"Dataset {dataset_name} already exists in project {project_name}."
                    " Set overwrite=True to overwrite or append=True to append."
                )
            else:
                cur_dataset = dataset
                break

    # either create new dataset or use existing one
    if cur_dataset is None:
        eval_dataset = client.projects.create_eval_dataset_for_project(
            project_id=project.id, name=dataset_name
        )
    else:
        eval_dataset = cur_dataset

    assert eval_dataset.id is not None

    # create questions
    if questions:
        questions = questions
    else:
        # download `LabelledRagDataset` from llama-hub
        assert llama_dataset_id is not None
        rag_dataset = _download_llama_dataset_from_hub(llama_dataset_id)
        questions = [example.query for example in rag_dataset[:]]

    eval_questions = client.evals.create_questions(
        dataset_id=eval_dataset.id,
        request=[EvalQuestionCreate(content=q) for q in questions],
    )

    assert len(eval_questions) == len(questions)
    print(f"Uploaded {len(questions)} questions to dataset {dataset_name}")
    return eval_dataset.id


def upload_eval_results(
    project_name: str, app_name: str, results: Dict[str, List[EvaluationResult]]
) -> None:
    """Upload the evaluation results to LlamaCloud.

    Args:
        project_name (str): The name of the project.
        app_name (str): The name of the app.
        results (Dict[str, List[EvaluationResult]]):
            The evaluation results, a mapping of metric name to a list of EvaluationResult objects.

    Examples:
        ```python
        from llama_index.core.evaluation.eval_utils import upload_eval_results

        result = evaluator.evaluate(...)
        upload_eval_results(
            project_name="my_project",
            app_name="my_app",
            results={"evaluator_name": [result]}
        )
        ```
    """
    from llama_cloud import ProjectCreate

    client = get_client()

    project = client.projects.upsert_project(request=ProjectCreate(name=project_name))
    assert project.id is not None

    client.projects.create_local_eval_set_for_project(
        project_id=project.id,
        app_name=app_name,
        results=results,
    )

    for key, val in results.items():
        print(
            f"Uploaded {len(val)} results for metric {key} under project {project_name}/{app_name}."
        )


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
