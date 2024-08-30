from typing import List, Optional

from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.bridge.pydantic import ConfigDict


class ReRankStartEvent(BaseEvent):
    """ReRankStartEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
        nodes (List[NodeWithScore]): List of nodes with scores.
        top_n (int): Number of nodes to return after rerank.
        model_name (str): Name of the model used for reranking.
    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    query: Optional[QueryType]
    nodes: List[NodeWithScore]
    top_n: int
    model_name: str

    @classmethod
    def class_name(cls):
        """Class name."""
        return "ReRankStartEvent"


class ReRankEndEvent(BaseEvent):
    """ReRankEndEvent.

    Args:
        nodes (List[NodeWithScore]): List of returned nodes after rerank.
    """

    nodes: List[NodeWithScore]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "ReRankEndEvent"
