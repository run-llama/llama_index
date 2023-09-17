import asyncio
from typing import Any, Dict, List, Optional, Tuple

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.response.schema import Response, RESPONSE_TYPE


class BatchEvalRunner:
    def __init__(
        self,
        evaluators: Dict[str, BaseEvaluator],
        workers: int = 2,
    ):
        self.evaluators = evaluators
        self.workers = workers

    async def aevaluate_queries(
        self,
        query_engine: BaseQueryEngine,
        queries: Optional[List[str]] = None,
        query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[EvaluationResult]]:
        if queries is None:
            raise ValueError("`queries` must be provided")
        query_kwargs = query_kwargs or {}

        semaphore = asyncio.Semaphore(self.workers)

        async def response_worker(
            query_engine: BaseQueryEngine,
            query: str,
        ) -> RESPONSE_TYPE:
            async with semaphore:
                return await query_engine.aquery(query)

        async def eval_worker(
            evaluator: BaseEvaluator,
            evaluator_name: str,
            query: str,
            response: Response,
            evaluator_kwargs: Dict[str, Any],
        ) -> Tuple[str, EvaluationResult]:
            async with semaphore:
                return (
                    evaluator_name,
                    await evaluator.aevaluate_response(
                        query=query, response=response, **evaluator_kwargs
                    ),
                )

        # gather responses
        response_jobs = []
        for query in queries:
            response_jobs.append(response_worker(query_engine, query))
        responses = await asyncio.gather(*response_jobs)

        # run evaluations
        eval_jobs = []
        for query, response in zip(queries, responses):
            for name, evaluator in self.evaluators.items():
                eval_jobs.append(
                    eval_worker(
                        evaluator, name, query, response, query_kwargs.get(query, {})
                    )
                )
        results = await asyncio.gather(*eval_jobs)

        # Format results
        results_dict: Dict[str, List[EvaluationResult]] = {
            name: [] for name in self.evaluators.keys()
        }
        for name, result in results:
            results_dict[name].append(result)

        return results_dict

    def evaluate_queries(
        self,
        query_engine: BaseQueryEngine,
        queries: Optional[List[str]] = None,
        query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[EvaluationResult]]:
        return asyncio.run(
            self.aevaluate_queries(
                query_engine=query_engine, queries=queries, query_kwargs=query_kwargs
            )
        )
