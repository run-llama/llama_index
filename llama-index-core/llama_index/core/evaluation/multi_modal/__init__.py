"""Multi-Modal Evaluation Modules."""

from llama_index.core.evaluation.multi_modal.faithfulness import (
    MultiModalFaithfulnessEvaluator,
)
from llama_index.core.evaluation.multi_modal.relevancy import (
    MultiModalRelevancyEvaluator,
)

__all__ = ["MultiModalRelevancyEvaluator", "MultiModalFaithfulnessEvaluator"]
