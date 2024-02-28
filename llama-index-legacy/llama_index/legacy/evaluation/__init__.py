"""Evaluation modules."""

from llama_index.legacy.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.legacy.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
)
from llama_index.legacy.evaluation.batch_runner import BatchEvalRunner
from llama_index.legacy.evaluation.context_relevancy import ContextRelevancyEvaluator
from llama_index.legacy.evaluation.correctness import CorrectnessEvaluator
from llama_index.legacy.evaluation.dataset_generation import (
    DatasetGenerator,
    QueryResponseDataset,
)
from llama_index.legacy.evaluation.faithfulness import (
    FaithfulnessEvaluator,
    ResponseEvaluator,
)
from llama_index.legacy.evaluation.guideline import GuidelineEvaluator
from llama_index.legacy.evaluation.notebook_utils import get_retrieval_results_df
from llama_index.legacy.evaluation.pairwise import PairwiseComparisonEvaluator
from llama_index.legacy.evaluation.relevancy import (
    QueryResponseEvaluator,
    RelevancyEvaluator,
)
from llama_index.legacy.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalResult,
)
from llama_index.legacy.evaluation.retrieval.evaluator import (
    MultiModalRetrieverEvaluator,
    RetrieverEvaluator,
)
from llama_index.legacy.evaluation.retrieval.metrics import (
    MRR,
    HitRate,
    RetrievalMetricResult,
    resolve_metrics,
)
from llama_index.legacy.evaluation.semantic_similarity import (
    SemanticSimilarityEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.answer_consistency import (
    AnswerConsistencyEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.answer_consistency_binary import (
    AnswerConsistencyBinaryEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.answer_similarity import (
    AnswerSimilarityEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.augmentation_accuracy import (
    AugmentationAccuracyEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.augmentation_precision import (
    AugmentationPrecisionEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.retrieval_precision import (
    RetrievalPrecisionEvaluator,
)
from llama_index.legacy.evaluation.tonic_validate.tonic_validate_evaluator import (
    TonicValidateEvaluator,
)

# import dataset generation too
from llama_index.legacy.finetuning.embeddings.common import (
    EmbeddingQAFinetuneDataset,
    generate_qa_embedding_pairs,
)

# aliases for generate_qa_embedding_pairs
generate_question_context_pairs = generate_qa_embedding_pairs
LabelledQADataset = EmbeddingQAFinetuneDataset

__all__ = [
    "BaseEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextRelevancyEvaluator",
    "EvaluationResult",
    "FaithfulnessEvaluator",
    "RelevancyEvaluator",
    "RelevanceEvaluator",
    "DatasetGenerator",
    "QueryResponseDataset",
    "GuidelineEvaluator",
    "CorrectnessEvaluator",
    "SemanticSimilarityEvaluator",
    "PairwiseComparisonEvaluator",
    "BatchEvalRunner",
    # legacy: kept for backward compatibility
    "QueryResponseEvaluator",
    "ResponseEvaluator",
    # retrieval
    "generate_qa_embedding_pairs",
    "generate_question_context_pairs",
    "EmbeddingQAFinetuneDataset",
    "BaseRetrievalEvaluator",
    "RetrievalEvalResult",
    "RetrieverEvaluator",
    "MultiModalRetrieverEvaluator",
    "RetrievalMetricResult",
    "resolve_metrics",
    "HitRate",
    "MRR",
    "get_retrieval_results_df",
    "LabelledQADataset",
    # tonic_validate evaluators
    "AnswerConsistencyEvaluator",
    "AnswerConsistencyBinaryEvaluator",
    "AnswerSimilarityEvaluator",
    "AugmentationAccuracyEvaluator",
    "AugmentationPrecisionEvaluator",
    "RetrievalPrecisionEvaluator",
    "TonicValidateEvaluator",
]
