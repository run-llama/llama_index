import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from llama_index.core import BaseQueryEngine
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.evaluation.eval_utils import asyncio_module
from llama_index.response.schema import RESPONSE_TYPE, Response


async def eval_response_worker(
    semaphore: asyncio.Semaphore,
    evaluator: BaseEvaluator,
    evaluator_name: str,
    query: Optional[str] = None,
    response: Optional[Response] = None,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, EvaluationResult]:
    """Get aevaluate_response tasks with semaphore."""
    eval_kwargs = eval_kwargs or {}
    async with semaphore:
        return (
            evaluator_name,
            await evaluator.aevaluate_response(
                query=query, response=response, **eval_kwargs
            ),
        )


async def eval_worker(
    semaphore: asyncio.Semaphore,
    evaluator: BaseEvaluator,
    evaluator_name: str,
    query: Optional[str] = None,
    response_str: Optional[str] = None,
    contexts: Optional[Sequence[str]] = None,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, EvaluationResult]:
    """Get aevaluate tasks with semaphore."""
    eval_kwargs = eval_kwargs or {}
    async with semaphore:
        return (
            evaluator_name,
            await evaluator.aevaluate(
                query=query, response=response_str, contexts=contexts, **eval_kwargs
            ),
        )


async def response_worker(
    semaphore: asyncio.Semaphore,
    query_engine: BaseQueryEngine,
    query: str,
) -> RESPONSE_TYPE:
    """Get aquery tasks with semaphore."""
    async with semaphore:
        return await query_engine.aquery(query)


class BatchEvalRunner:
    """Batch evaluation runner.

    Args:
        evaluators (Dict[str, BaseEvaluator]): Dictionary of evaluators.
        workers (int): Number of workers to use for parallelization.
            Defaults to 2.
        show_progress (bool): Whether to show progress bars. Defaults to False.

    """

    def __init__(
        self,
        evaluators: Dict[str, BaseEvaluator],
        workers: int = 2,
        show_progress: bool = False,
    ):
        self.evaluators = evaluators
        self.workers = workers
        self.semaphore = asyncio.Semaphore(self.workers)
        self.show_progress = show_progress
        self.asyncio_mod = asyncio_module(show_progress=self.show_progress)

    def _format_results(
        self, results: List[EvaluationResult]
    ) -> Dict[str, List[EvaluationResult]]:
        """Format results."""
        # Format results
        results_dict: Dict[str, List[EvaluationResult]] = {
            name: [] for name in self.evaluators
        }
        for name, result in results:
            results_dict[name].append(result)

        return results_dict

    def _validate_and_clean_inputs(
        self,
        *inputs_list: Any,
    ) -> List[Any]:
        """Validate and clean input lists.

        Enforce that at least one of the inputs is not None.
        Make sure that all inputs have the same length.
        Make sure that None inputs are replaced with [None] * len(inputs).

        """
        assert len(inputs_list) > 0
        # first, make sure at least one of queries or response_strs is not None
        input_len: Optional[int] = None
        for inputs in inputs_list:
            if inputs is not None:
                input_len = len(inputs)
                break
        if input_len is None:
            raise ValueError("At least one item in inputs_list must be provided.")

        new_inputs_list = []
        for inputs in inputs_list:
            if inputs is None:
                new_inputs_list.append([None] * input_len)
            else:
                if len(inputs) != input_len:
                    raise ValueError("All inputs must have the same length.")
                new_inputs_list.append(inputs)
        return new_inputs_list

    def _get_eval_kwargs(
        self, eval_kwargs_lists: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Get eval kwargs from eval_kwargs_lists at a given idx.

        Since eval_kwargs_lists is a dict of lists, we need to get the
        value at idx for each key.

        """
        return {k: v[idx] for k, v in eval_kwargs_lists.items()}

    async def aevaluate_response_strs(
        self,
        queries: Optional[List[str]] = None,
        response_strs: Optional[List[str]] = None,
        contexts_list: Optional[List[List[str]]] = None,
        **eval_kwargs_lists: List,
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate query, response pairs.

        This evaluates queries, responses, contexts as string inputs.
        Can supply additional kwargs to the evaluator in eval_kwargs_lists.

        Args:
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            response_strs (Optional[List[str]]): List of response strings.
                Defaults to None.
            contexts_list (Optional[List[List[str]]]): List of context lists.
                Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        queries, response_strs, contexts_list = self._validate_and_clean_inputs(
            queries, response_strs, contexts_list
        )
        for k in eval_kwargs_lists:
            v = eval_kwargs_lists[k]
            if not isinstance(v, list):
                raise ValueError(
                    f"Each value in eval_kwargs must be a list. Got {k}: {v}"
                )
            eval_kwargs_lists[k] = self._validate_and_clean_inputs(v)[0]

        # run evaluations
        eval_jobs = []
        for idx, query in enumerate(cast(List[str], queries)):
            response_str = cast(List, response_strs)[idx]
            contexts = cast(List, contexts_list)[idx]
            eval_kwargs = self._get_eval_kwargs(eval_kwargs_lists, idx)
            for name, evaluator in self.evaluators.items():
                eval_jobs.append(
                    eval_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        response_str=response_str,
                        contexts=contexts,
                        eval_kwargs=eval_kwargs,
                    )
                )
        results = await self.asyncio_mod.gather(*eval_jobs)

        # Format results
        return self._format_results(results)

    async def aevaluate_responses(
        self,
        queries: Optional[List[str]] = None,
        responses: Optional[List[Response]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate query, response pairs.

        This evaluates queries and response objects.

        Args:
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            responses (Optional[List[Response]]): List of response objects.
                Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        queries, responses = self._validate_and_clean_inputs(queries, responses)
        for k in eval_kwargs_lists:
            v = eval_kwargs_lists[k]
            if not isinstance(v, list):
                raise ValueError(
                    f"Each value in eval_kwargs must be a list. Got {k}: {v}"
                )
            eval_kwargs_lists[k] = self._validate_and_clean_inputs(v)[0]

        # run evaluations
        eval_jobs = []
        for idx, query in enumerate(cast(List[str], queries)):
            response = cast(List, responses)[idx]
            eval_kwargs = self._get_eval_kwargs(eval_kwargs_lists, idx)
            for name, evaluator in self.evaluators.items():
                eval_jobs.append(
                    eval_response_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        response=response,
                        eval_kwargs=eval_kwargs,
                    )
                )
        results = await self.asyncio_mod.gather(*eval_jobs)

        # Format results
        return self._format_results(results)

    async def aevaluate_queries(
        self,
        query_engine: BaseQueryEngine,
        queries: Optional[List[str]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate queries.

        Args:
            query_engine (BaseQueryEngine): Query engine.
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        if queries is None:
            raise ValueError("`queries` must be provided")

        # gather responses
        response_jobs = []
        for query in queries:
            response_jobs.append(response_worker(self.semaphore, query_engine, query))
        responses = await self.asyncio_mod.gather(*response_jobs)

        return await self.aevaluate_responses(
            queries=queries,
            responses=responses,
            **eval_kwargs_lists,
        )

    def evaluate_response_strs(
        self,
        queries: Optional[List[str]] = None,
        response_strs: Optional[List[str]] = None,
        contexts_list: Optional[List[List[str]]] = None,
        **eval_kwargs_lists: List,
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate query, response pairs.

        Sync version of aevaluate_response_strs.

        """
        return asyncio.run(
            self.aevaluate_response_strs(
                queries=queries,
                response_strs=response_strs,
                contexts_list=contexts_list,
                **eval_kwargs_lists,
            )
        )

    def evaluate_responses(
        self,
        queries: Optional[List[str]] = None,
        responses: Optional[List[Response]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate query, response objs.

        Sync version of aevaluate_responses.

        """
        return asyncio.run(
            self.aevaluate_responses(
                queries=queries,
                responses=responses,
                **eval_kwargs_lists,
            )
        )

    def evaluate_queries(
        self,
        query_engine: BaseQueryEngine,
        queries: Optional[List[str]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate queries.

        Sync version of aevaluate_queries.

        """
        return asyncio.run(
            self.aevaluate_queries(
                query_engine=query_engine,
                queries=queries,
                **eval_kwargs_lists,
            )
        )
