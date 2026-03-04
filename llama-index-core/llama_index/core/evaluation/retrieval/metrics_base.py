from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict


class RetrievalMetricResult(BaseModel):
    """
    Metric result.

    Attributes:
        score (float): Score for the metric
        metadata (Dict[str, Any]): Metadata for the metric result

    """

    score: float = Field(..., description="Score for the metric")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata for the metric result"
    )

    def __str__(self) -> str:
        """String representation."""
        return f"Score: {self.score}\nMetadata: {self.metadata}"

    def __float__(self) -> float:
        """Float representation."""
        return self.score


class BaseRetrievalMetric(BaseModel, ABC):
    """Base class for retrieval metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric_name: ClassVar[str]

    @abstractmethod
    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """
        Compute metric.

        Args:
            query (Optional[str]): Query string
            expected_ids (Optional[List[str]]): Expected ids
            retrieved_ids (Optional[List[str]]): Retrieved ids
            **kwargs: Additional keyword arguments

        """
