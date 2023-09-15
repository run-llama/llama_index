"""Evaluation modules."""

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.evaluation.correctness import CorrectnessEvaluator
from llama_index.evaluation.dataset_generation import DatasetGenerator
from llama_index.evaluation.faithfulness import FaithfulnessEvaluator, ResponseEvaluator
from llama_index.evaluation.guideline import GuidelineEvaluator
from llama_index.evaluation.relevancy import QueryResponseEvaluator, RelevancyEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "FaithfulnessEvaluator",
    "RelevancyEvaluator",
    "RelevanceEvaluator",
    "DatasetGenerator",
    "GuidelineEvaluator",
    "CorrectnessEvaluator",
    # legacy: kept for backward compatibility
    "QueryResponseEvaluator",
    "ResponseEvaluator",
]
