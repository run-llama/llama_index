"""Evaluation modules."""

from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.core.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
)
from llama_index.core.evaluation.batch_runner import BatchEvalRunner
from llama_index.core.evaluation.context_relevancy import ContextRelevancyEvaluator
from llama_index.core.evaluation.correctness import CorrectnessEvaluator
from llama_index.core.evaluation.dataset_generation import (
    DatasetGenerator,
    QueryResponseDataset,
)
from llama_index.core.evaluation.faithfulness import (
    FaithfulnessEvaluator,
    ResponseEvaluator,
)
from llama_index.core.evaluation.guideline import GuidelineEvaluator
from llama_index.core.evaluation.notebook_utils import get_retrieval_results_df
from llama_index.core.evaluation.pairwise import PairwiseComparisonEvaluator
from llama_index.core.evaluation.relevancy import (
    QueryResponseEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalResult,
)
from llama_index.core.evaluation.retrieval.evaluator import (
    MultiModalRetrieverEvaluator,
    RetrieverEvaluator,
)
from llama_index.core.evaluation.retrieval.metrics import (
    MRR,
    HitRate,
    RetrievalMetricResult,
    resolve_metrics,
)
from llama_index.core.evaluation.semantic_similarity import (
    SemanticSimilarityEvaluator,
)

# import dataset generation too
from llama_index.core.llama_dataset.legacy.embedding import (
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
]
