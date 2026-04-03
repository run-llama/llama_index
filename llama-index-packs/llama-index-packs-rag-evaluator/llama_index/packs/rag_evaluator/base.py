import asyncio
import json
import time
import warnings
from collections import deque
from typing import Any, List, Optional
import os
from pathlib import Path

import pandas as pd
import tqdm
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    EvaluationResult,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)
from llama_index.core.evaluation.notebook_utils import (
    get_eval_results_df,
)
from llama_index.core.llama_dataset import BaseLlamaDataset, BaseLlamaPredictionDataset
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from openai import RateLimitError
from tqdm.asyncio import tqdm_asyncio


class RagEvaluatorPack(BaseLlamaPack):
    """
    A pack for performing evaluation with your own RAG pipeline.

    Args:
        query_engine: The RAG pipeline to evaluate.
        rag_dataset: The BaseLlamaDataset to evaluate on.
        judge_llm: The LLM to use as the evaluator.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        rag_dataset: BaseLlamaDataset,
        judge_llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        show_progress: bool = True,
        result_path: Optional[str] = None,
    ):
        self.query_engine = query_engine
        self.rag_dataset = rag_dataset
        self._num_examples = len(self.rag_dataset.examples)
        if judge_llm is None:
            self.judge_llm = OpenAI(temperature=0, model="gpt-4-1106-preview")
        else:
            assert isinstance(judge_llm, LLM)
            self.judge_llm = judge_llm

        self.embed_model = embed_model or Settings.embed_model
        self.show_progress = show_progress
        self.evals = {
            "correctness": [],
            "relevancy": [],
            "faithfulness": [],
            "context_similarity": [],
        }
        self.eval_queue = deque(range(len(rag_dataset.examples)))
        self.prediction_dataset = None
        if result_path is None:
            self.result_path = Path.cwd()
        else:
            self.result_path = Path(result_path)
            if not self.result_path.is_absolute():
                self.result_path = Path.cwd() / self.result_path

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    async def _amake_predictions(
        self,
        batch_size: int = 20,
        sleep_time_in_seconds: int = 1,
    ):
        """Async make predictions with query engine."""
        self.prediction_dataset: BaseLlamaPredictionDataset = (
            await self.rag_dataset.amake_predictions_with(
                self.query_engine,
                show_progress=self.show_progress,
                batch_size=batch_size,
                sleep_time_in_seconds=sleep_time_in_seconds,
            )
        )

    def _make_predictions(
        self,
        batch_size: int = 20,
        sleep_time_in_seconds: int = 1,
    ):
        """Sync make predictions with query engine."""
        self.prediction_dataset: BaseLlamaPredictionDataset = (
            self.rag_dataset.make_predictions_with(
                self.query_engine,
                show_progress=self.show_progress,
                batch_size=batch_size,
                sleep_time_in_seconds=sleep_time_in_seconds,
            )
        )

    def _prepare_judges(self):
        """Construct the evaluators."""
        judges = {}
        judges["correctness"] = CorrectnessEvaluator(
            llm=self.judge_llm,
        )
        judges["relevancy"] = RelevancyEvaluator(
            llm=self.judge_llm,
        )
        judges["faithfulness"] = FaithfulnessEvaluator(
            llm=self.judge_llm,
        )
        judges["semantic_similarity"] = SemanticSimilarityEvaluator(
            embed_model=self.embed_model
        )
        return judges

    async def _areturn_null_eval_result(self, query) -> EvaluationResult:
        """
        A dummy async method that returns None.

        NOTE: this is used to handle case when creating async tasks for evaluating
        predictions where contexts do not exist.
        """
        return EvaluationResult(
            query=query,
        )

    def _return_null_eval_result(self, query) -> EvaluationResult:
        """
        A dummy async method that returns None.

        NOTE: this is used to handle case when creating async tasks for evaluating
        predictions where contexts do not exist.
        """
        return EvaluationResult(
            query=query,
        )

    def _create_async_evaluate_example_prediction_tasks(
        self, judges, example, prediction, sleep_time_in_seconds
    ):
        """Collect the co-routines."""
        correctness_task = judges["correctness"].aevaluate(
            query=example.query,
            response=prediction.response,
            reference=example.reference_answer,
            sleep_time_in_seconds=sleep_time_in_seconds,
        )

        relevancy_task = judges["relevancy"].aevaluate(
            query=example.query,
            response=prediction.response,
            contexts=prediction.contexts,
            sleep_time_in_seconds=sleep_time_in_seconds,
        )

        faithfulness_task = judges["faithfulness"].aevaluate(
            query=example.query,
            response=prediction.response,
            contexts=prediction.contexts,
            sleep_time_in_seconds=sleep_time_in_seconds,
        )

        if example.reference_contexts and prediction.contexts:
            semantic_similarity_task = judges["semantic_similarity"].aevaluate(
                query=example.query,
                response="\n".join(prediction.contexts),
                reference="\n".join(example.reference_contexts),
            )
        else:
            semantic_similarity_task = self._areturn_null_eval_result(
                query=example.query
            )

        return (
            correctness_task,
            relevancy_task,
            faithfulness_task,
            semantic_similarity_task,
        )

    def _evaluate_example_prediction(self, judges, example, prediction):
        """Collect the co-routines."""
        correctness_result = judges["correctness"].evaluate(
            query=example.query,
            response=prediction.response,
            reference=example.reference_answer,
        )

        relevancy_result = judges["relevancy"].evaluate(
            query=example.query,
            response=prediction.response,
            contexts=prediction.contexts,
        )

        faithfulness_result = judges["faithfulness"].evaluate(
            query=example.query,
            response=prediction.response,
            contexts=prediction.contexts,
        )

        if example.reference_contexts and prediction.contexts:
            semantic_similarity_result = judges["semantic_similarity"].evaluate(
                query=example.query,
                response="\n".join(prediction.contexts),
                reference="\n".join(example.reference_contexts),
            )
        else:
            semantic_similarity_result = self._return_null_eval_result(
                query=example.query
            )

        return (
            correctness_result,
            relevancy_result,
            faithfulness_result,
            semantic_similarity_result,
        )

    def _save_evaluations(self):
        """Save evaluation json object."""
        # saving evaluations
        evaluations_objects = {
            "context_similarity": [e.dict() for e in self.evals["context_similarity"]],
            "correctness": [e.dict() for e in self.evals["correctness"]],
            "faithfulness": [e.dict() for e in self.evals["faithfulness"]],
            "relevancy": [e.dict() for e in self.evals["relevancy"]],
        }

        with open(
            os.path.join(self.result_path, "_evaluations.json"), "w"
        ) as json_file:
            json.dump(evaluations_objects, json_file)

    def _prepare_and_save_benchmark_results(self):
        """Get mean score across all of the evaluated examples-predictions."""
        _, mean_correctness_df = get_eval_results_df(
            ["base_rag"] * len(self.evals["correctness"]),
            self.evals["correctness"],
            metric="correctness",
        )
        _, mean_relevancy_df = get_eval_results_df(
            ["base_rag"] * len(self.evals["relevancy"]),
            self.evals["relevancy"],
            metric="relevancy",
        )
        _, mean_faithfulness_df = get_eval_results_df(
            ["base_rag"] * len(self.evals["faithfulness"]),
            self.evals["faithfulness"],
            metric="faithfulness",
        )
        _, mean_context_similarity_df = get_eval_results_df(
            ["base_rag"] * len(self.evals["context_similarity"]),
            self.evals["context_similarity"],
            metric="context_similarity",
        )

        mean_scores_df = pd.concat(
            [
                mean_correctness_df.reset_index(),
                mean_relevancy_df.reset_index(),
                mean_faithfulness_df.reset_index(),
                mean_context_similarity_df.reset_index(),
            ],
            axis=0,
            ignore_index=True,
        )
        mean_scores_df = mean_scores_df.set_index("index")
        mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])

        # save mean_scores_df
        mean_scores_df.to_csv(os.path.join(self.result_path, "benchmark.csv"))
        return mean_scores_df

    def _make_evaluations(
        self,
        batch_size,
        sleep_time_in_seconds,
    ):
        """Sync make evaluations."""
        judges = self._prepare_judges()

        start_ix = self.eval_queue[0]
        for batch in self._batch_examples_and_preds(
            self.rag_dataset.examples,
            self.prediction_dataset.predictions,
            batch_size=batch_size,
            start_position=start_ix,
        ):
            examples, predictions = batch
            for example, prediction in tqdm.tqdm(zip(examples, predictions)):
                (
                    correctness_result,
                    relevancy_result,
                    faithfulness_result,
                    semantic_similarity_result,
                ) = self._evaluate_example_prediction(
                    judges=judges, example=example, prediction=prediction
                )

                self.evals["correctness"].append(correctness_result)
                self.evals["relevancy"].append(relevancy_result)
                self.evals["faithfulness"].append(faithfulness_result)
                self.evals["context_similarity"].append(semantic_similarity_result)
            time.sleep(sleep_time_in_seconds)

        self._save_evaluations()
        return self._prepare_and_save_benchmark_results()

    def _batch_examples_and_preds(
        self,
        examples: List[Any],
        predictions: List[Any],
        batch_size: int = 10,
        start_position: int = 0,
    ):
        """Batches examples and predictions with a given batch_size."""
        assert self._num_examples == len(predictions)
        for ndx in range(start_position, self._num_examples, batch_size):
            yield (
                examples[ndx : min(ndx + batch_size, self._num_examples)],
                predictions[ndx : min(ndx + batch_size, self._num_examples)],
            )

    async def _amake_evaluations(self, batch_size, sleep_time_in_seconds):
        """Async make evaluations."""
        judges = self._prepare_judges()

        ix = self.eval_queue[0]
        batch_iterator = self._batch_examples_and_preds(
            self.rag_dataset.examples,
            self.prediction_dataset.predictions,
            batch_size=batch_size,
            start_position=ix,
        )
        total_batches = (self._num_examples - ix + 1) / batch_size + (
            (self._num_examples - ix + 1) % batch_size != 0
        )
        if self.show_progress:
            batch_iterator = tqdm_asyncio(
                batch_iterator,
                desc="Batch processing of evaluations",
                total=total_batches,
            )

        for batch in batch_iterator:
            examples, predictions = batch
            tasks = []
            for example, prediction in zip(examples, predictions):
                (
                    correctness_task,
                    relevancy_task,
                    faithfulness_task,
                    semantic_similarity_task,
                ) = self._create_async_evaluate_example_prediction_tasks(
                    judges=judges,
                    example=example,
                    prediction=prediction,
                    sleep_time_in_seconds=sleep_time_in_seconds,
                )

                tasks += [
                    correctness_task,
                    relevancy_task,
                    faithfulness_task,
                    semantic_similarity_task,
                ]

            # do this in batches to avoid RateLimitError
            try:
                eval_results: List[EvaluationResult] = await asyncio.gather(*tasks)
            except RateLimitError as err:
                if self.show_progress:
                    batch_iterator.close()
                raise ValueError(
                    "You've hit rate limits on your OpenAI subscription. This"
                    " `RagEvaluatorPack` maintains state of evaluations. Simply"
                    " re-invoke .arun() in order to continue from where you left"
                    " off."
                ) from err
            # store in memory
            # since final result of eval_results respects order of inputs
            # just take appropriate slices
            self.evals["correctness"] += eval_results[::4]
            self.evals["relevancy"] += eval_results[1::4]
            self.evals["faithfulness"] += eval_results[2::4]
            self.evals["context_similarity"] += eval_results[3::4]
            # update queue
            for _ in range(batch_size):
                if self.eval_queue:
                    self.eval_queue.popleft()
            ix += 1
            if self.show_progress:
                batch_iterator.update()
                batch_iterator.refresh()

        self._save_evaluations()
        return self._prepare_and_save_benchmark_results()

    def run(self, batch_size: int = 10, sleep_time_in_seconds: int = 1):
        if batch_size > 10:
            warnings.warn(
                "You've set a large batch_size (>10). If using OpenAI GPT-4 as "
                " `judge_llm` (which is the default judge_llm),"
                " you may experience a RateLimitError. Previous successful eval "
                " responses are cached per batch. So hitting a RateLimitError"
                " would mean you'd lose all of the current batches successful "
                " GPT-4 calls."
            )
        if self.prediction_dataset is None:
            self._make_predictions(batch_size, sleep_time_in_seconds)

        # evaluate predictions
        eval_sleep_time_in_seconds = (
            sleep_time_in_seconds * 2
        )  # since we make 3 evaluator llm calls
        eval_batch_size = int(max(batch_size / 4, 1))
        return self._make_evaluations(
            batch_size=eval_batch_size, sleep_time_in_seconds=eval_sleep_time_in_seconds
        )

    async def arun(
        self,
        batch_size: int = 10,
        sleep_time_in_seconds: int = 1,
    ):
        if batch_size > 10:
            warnings.warn(
                "You've set a large batch_size (>10). If using OpenAI GPT-4 as "
                " `judge_llm` (which is the default judge_llm),"
                " you may experience a RateLimitError. Previous successful eval "
                " responses are cached per batch. So hitting a RateLimitError"
                " would mean you'd lose all of the current batches successful "
                " GPT-4 calls."
            )

        # make predictions
        if self.prediction_dataset is None:
            await self._amake_predictions(batch_size, sleep_time_in_seconds)

        # evaluate predictions
        eval_sleep_time_in_seconds = (
            sleep_time_in_seconds * 2
        )  # since we make 3 evaluator llm calls and default is gpt-4
        # which is heavily rate-limited
        eval_batch_size = int(max(batch_size / 4, 1))
        return await self._amake_evaluations(
            batch_size=eval_batch_size, sleep_time_in_seconds=eval_sleep_time_in_seconds
        )
