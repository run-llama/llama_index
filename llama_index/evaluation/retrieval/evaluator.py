"""Retrieval evaluators."""

from typing import Any, List, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.core import BaseRetriever
from llama_index.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
)
from llama_index.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
)
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.schema import BaseNode, ImageNode


class RetrieverEvaluator(BaseRetrievalEvaluator):
    """Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.

    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")

    def __init__(
        self,
        metrics: Sequence[BaseRetrievalMetric],
        retriever: BaseRetriever,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metrics=metrics, retriever=retriever, **kwargs)

    async def _aget_retrieved_ids(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> List[str]:
        """Get retrieved ids."""
        retrieved_nodes = await self.retriever.aretrieve(query)
        return [node.node.node_id for node in retrieved_nodes]


class MultiModalRetrieverEvaluator(BaseRetrievalEvaluator):
    """Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.

    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")

    def __init__(
        self,
        metrics: Sequence[BaseRetrievalMetric],
        retriever: BaseRetriever,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metrics=metrics, retriever=retriever, **kwargs)

    async def _aget_retrieved_ids(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> List[str]:
        """Get retrieved ids."""
        retrieved_nodes = await self.retriever.aretrieve(query)
        image_nodes: List[ImageNode] = []
        text_nodes: List[BaseNode] = []

        for scored_node in retrieved_nodes:
            node = scored_node.node
            if isinstance(node, ImageNode):
                image_nodes.append(node)
            if node.text:
                text_nodes.append(node)

        if mode == "text":
            return [node.node_id for node in text_nodes]
        elif mode == "image":
            return [node.node_id for node in image_nodes]
        else:
            raise ValueError("Unsupported mode.")
