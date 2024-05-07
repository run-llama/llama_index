from typing import List
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.schema import QueryType, NodeWithScore


class RetrievalStartEvent(BaseEvent):
    """RetrievalStartEvent.

    Args:
        str_or_query_bundle (QueryType): Query bundle.
    """

    str_or_query_bundle: QueryType

    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalStartEvent"


class RetrievalEndEvent(BaseEvent):
    """RetrievalEndEvent.

    Args:
        str_or_query_bundle (QueryType): Query bundle.
        nodes (List[NodeWithScore]): List of nodes with scores.
    """

    str_or_query_bundle: QueryType
    nodes: List[NodeWithScore]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalEndEvent"
