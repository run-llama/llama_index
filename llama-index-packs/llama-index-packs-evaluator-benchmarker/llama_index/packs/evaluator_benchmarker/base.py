import warnings
from typing import Union

import numpy as np
import pandas as pd
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llama_dataset.evaluator_evaluation import (
    EvaluatorPredictionDataset,
    LabelledEvaluatorDataset,
    LabelledPairwiseEvaluatorDataset,
    PairwiseEvaluatorPredictionDataset,
)
from llama_index.core.llama_pack.base import BaseLlamaPack


class EvaluatorBenchmarkerPack(BaseLlamaPack):
    """
    A pack for benchmarking/evaluating your own evaluator.

    Args:
        evaluator (BaseEvaluator): The evaluator to evaluate/benchmark.
        eval_dataset (LabelledEvaluatorDataset | LabelledPairwiseEvaluatorDataset): The
            labelled evaluation dataset to run benchmarks against.

    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        eval_dataset: Union[LabelledEvaluatorDataset, LabelledPairwiseEvaluatorDataset],
        show_progress: bool = True,
    ):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self._num_examples = len(self.eval_dataset.examples)
        self.show_progress = show_progress
        self.prediction_dataset = None

    async def _amake_predictions(
        self,
        batch_size: int = 20,
        sleep_time_in_seconds: int = 1,
    ):
        """Async make predictions with evaluator."""
        self.prediction_dataset: Union[
            EvaluatorPredictionDataset, PairwiseEvaluatorPredictionDataset
        ] = await self.eval_dataset.amake_predictions_with(
            predictor=self.evaluator,
            show_progress=self.show_progress,
            batch_size=batch_size,
            sleep_time_in_seconds=sleep_time_in_seconds,
        )

    def make_predictions(self, batch_size: int = 20, sleep_time_in_seconds: int = 1):
        """Sync make predictions with evaluator."""
        self.prediction_dataset: Union[
            EvaluatorPredictionDataset, PairwiseEvaluatorPredictionDataset
        ] = self.eval_dataset.make_predictions_with(
            predictor=self.evaluator,
            show_progress=self.show_progress,
            batch_size=batch_size,
            sleep_time_in_seconds=sleep_time_in_seconds,
        )

    def _prepare_and_save_benchmark_results_pairwise_grading(self) -> pd.DataFrame:
        """Compute benchmark metrics for pairwise evaluation."""
        inconclusive_counts = 0
        agreements_with_ties = 0
        agreements_without_ties = 0
        ties = 0
        invalid_counts = 0
        for example, prediction in zip(
            self.eval_dataset[:], self.prediction_dataset[:]
        ):
            if prediction.invalid_prediction:
                invalid_counts += 1
                continue

            # don't count inconclusive results
            if prediction.evaluation_source == "neither":
                inconclusive_counts += 1
                continue

            if prediction.score == 0.5 or example.reference_score == 0.5:
                ties += 1
            else:
                agreements_without_ties += int(
                    example.reference_score == prediction.score
                )
            agreements_with_ties += int(example.reference_score == prediction.score)

        agreement_rate_with_ties = agreements_with_ties / (
            len(self.prediction_dataset[:]) - inconclusive_counts - invalid_counts
        )
        agreement_rate_without_ties = agreements_without_ties / (
            len(self.prediction_dataset[:])
            - inconclusive_counts
            - ties
            - invalid_counts
        )

        df_data = {
            "number_examples": [len(self.prediction_dataset[:])],
            "invalid_predictions": [invalid_counts],
            "inconclusives": [inconclusive_counts],
            "ties": [ties],
            "agreement_rate_with_ties": [agreement_rate_with_ties],
            "agreement_rate_without_ties": [agreement_rate_without_ties],
        }
        benchmark_df = pd.DataFrame(df_data)
        benchmark_df.to_csv("benchmark.csv")
        return benchmark_df

    def _prepare_and_save_benchmark_results_single_grading(self) -> pd.DataFrame:
        """Compute benchmark metrics for single grading evaluation."""
        invalid_counts = sum([p.invalid_prediction for p in self.prediction_dataset[:]])
        np_preds = np.array([p.score for p in self.prediction_dataset[:]])
        np_refs = np.array([e.reference_score for e in self.eval_dataset[:]])
        invalid_mask = ~np.array(
            [p.invalid_prediction for p in self.prediction_dataset[:]]
        )

        # metrics
        mae = np.mean(np.abs(np_preds[invalid_mask] - np_refs[invalid_mask]))
        corr = np.corrcoef(
            np_preds[invalid_mask].astype(float), np_refs[invalid_mask].astype(float)
        )[0, 1]
        hamming = np.sum(np_preds[invalid_mask] == np_refs[invalid_mask])

        df_data = {
            "number_examples": [len(self.prediction_dataset[:])],
            "invalid_predictions": [invalid_counts],
            "correlation": [corr],
            "mae": [mae],
            "hamming": [hamming],
        }
        benchmark_df = pd.DataFrame(df_data)
        benchmark_df.to_csv("benchmark.csv")
        return benchmark_df

    def _make_evaluations(self) -> pd.DataFrame:
        """Returns benchmark_df."""
        if isinstance(self.eval_dataset, LabelledPairwiseEvaluatorDataset):
            return self._prepare_and_save_benchmark_results_pairwise_grading()
        else:
            return self._prepare_and_save_benchmark_results_single_grading()

    async def arun(self, batch_size: int = 10, sleep_time_in_seconds: int = 1):
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

        # produce metrics
        return self._make_evaluations()
