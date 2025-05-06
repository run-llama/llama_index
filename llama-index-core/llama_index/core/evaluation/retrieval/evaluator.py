"""Retrieval evaluators."""

from typing import List, Optional, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
)
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import ImageNode, TextNode


class RetrieverEvaluator(BaseRetrievalEvaluator):
    """
    Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Post-processor to apply after retrieval.


    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")
    node_postprocessors: Optional[List[SerializeAsAny[BaseNodePostprocessor]]] = Field(
        default=None, description="Optional post-processor"
    )

    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        """Get retrieved ids and texts, potentially applying a post-processor."""
        retrieved_nodes = await self.retriever.aretrieve(query)

        if self.node_postprocessors:
            for node_postprocessor in self.node_postprocessors:
                retrieved_nodes = node_postprocessor.postprocess_nodes(
                    retrieved_nodes, query_str=query
                )

        return (
            [node.node.node_id for node in retrieved_nodes],
            [node.text for node in retrieved_nodes],
        )


class MultiModalRetrieverEvaluator(BaseRetrievalEvaluator):
    """
    Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Post-processor to apply after retrieval.

    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")
    node_postprocessors: Optional[List[SerializeAsAny[BaseNodePostprocessor]]] = Field(
        default=None, description="Optional post-processor"
    )

    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        """Get retrieved ids."""
        retrieved_nodes = await self.retriever.aretrieve(query)
        image_nodes: List[ImageNode] = []
        text_nodes: List[TextNode] = []

        if self.node_postprocessors:
            for node_postprocessor in self.node_postprocessors:
                retrieved_nodes = node_postprocessor.postprocess_nodes(
                    retrieved_nodes, query_str=query
                )

        for scored_node in retrieved_nodes:
            node = scored_node.node
            if isinstance(node, ImageNode):
                image_nodes.append(node)
            if isinstance(node, TextNode):
                text_nodes.append(node)

        if mode == "text":
            return (
                [node.node_id for node in text_nodes],
                [node.text for node in text_nodes],
            )
        elif mode == "image":
            return (
                [node.node_id for node in image_nodes],
                [node.text for node in image_nodes],
            )
        else:
            raise ValueError("Unsupported mode.")
