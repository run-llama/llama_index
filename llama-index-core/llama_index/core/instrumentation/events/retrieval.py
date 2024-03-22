from typing import List
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.schema import QueryType, NodeWithScore


class RetrievalStartEvent(BaseEvent):
    str_or_query_bundle: QueryType

    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalStartEvent"


class RetrievalEndEvent(BaseEvent):
    str_or_query_bundle: QueryType
    nodes: List[NodeWithScore]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalEndEvent"
