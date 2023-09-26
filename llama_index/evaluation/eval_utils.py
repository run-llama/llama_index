"""Get evaluation utils.

NOTE: These are beta functions, might change.

"""

from llama_index.indices.query.base import BaseQueryEngine
from llama_index.evaluation.base import EvaluationResult, BaseEvaluator
from llama_index.utils import get_tqdm_iterable
from typing import List, Any
import asyncio
import numpy as np


def asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio
        module = tqdm_asyncio
    else:
        module = asyncio

    return module


async def aget_responses(
    questions: List[str], 
    query_engine: BaseQueryEngine,
    show_progress: bool = False
) -> List[str]:
    """Get responses."""
    tasks = []
    for question in questions:
        tasks.append(query_engine.aquery(question))
    asyncio_module = asyncio_module(show_progress=show_progress)
    responses = await asyncio_module.gather(*tasks)
    return responses


def get_responses(
    *args: Any,
    **kwargs: Any,
) -> List[str]:
    """Get responses.

    Sync version of aget_responses.
    
    """
    responses = asyncio.run(aget_responses(*args, **kwargs))
    return responses
