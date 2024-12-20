from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryType


class QueryStartEvent(BaseEvent):
    """QueryStartEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
    """

    query: QueryType

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "QueryStartEvent"


class QueryEndEvent(BaseEvent):
    """QueryEndEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
        response (RESPONSE_TYPE): Response.
    """

    query: QueryType
    response: RESPONSE_TYPE

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "QueryEndEvent"
