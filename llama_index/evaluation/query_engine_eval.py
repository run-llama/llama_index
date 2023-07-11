from __future__ import annotations

from dataclasses import dataclass

from llama_index.async_utils import run_async_tasks
from llama_index.indices.base import ServiceContext
from llama_index.indices.query.base import BaseQueryEngine


@dataclass
class QueryEngineEvaluator:
    # add a Callback manager here
    service_context: ServiceContext

    def evaluate(self, query_engine: BaseQueryEngine, questions: list[str]):
        """
        Evaluate.
        """

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_relevancy, faithfulness
        except ImportError:
            raise ImportError(
                "Please install ragas via `pip install ragas` to use"
                " QueryEngineEvaluator"
            )
        # TODO: rate limit, error handling, retries
        responses = run_async_tasks([query_engine.aquery(q) for q in questions])

        answers = []
        contexts = []
        for r in responses:
            answers.append(r.response)
            contexts.append([c.node.get_content() for c in r.source_nodes])

        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
        )
        # TODO: create a wrapper around this result for llama_index
        result = evaluate(ds)

        return result
