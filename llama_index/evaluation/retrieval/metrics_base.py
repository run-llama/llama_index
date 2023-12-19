from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.bridge.pydantic import BaseModel, Field


class RetrievalMetricResult(BaseModel):
    """Metric result.

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

    metric_name: str

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
        """Compute metric.

        Args:
            query (Optional[str]): Query string
            expected_ids (Optional[List[str]]): Expected ids
            retrieved_ids (Optional[List[str]]): Retrieved ids
            **kwargs: Additional keyword arguments

        """

    class Config:
        arbitrary_types_allowed = True
