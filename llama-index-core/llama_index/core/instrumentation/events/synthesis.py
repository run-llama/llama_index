from typing import List

from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryType


class SynthesizeStartEvent(BaseEvent):
    """SynthesizeStartEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
    """

    query: QueryType

    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeStartEvent"


class SynthesizeEndEvent(BaseEvent):
    """SynthesizeEndEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
        response (RESPONSE_TYPE): Response.
    """

    query: QueryType
    response: RESPONSE_TYPE

    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeEndEvent"


class GetResponseStartEvent(BaseEvent):
    """GetResponseStartEvent.

    Args:
        query_str (str): Query string.
        text_chunks (List[str]): List of text chunks.
    """

    query_str: str
    text_chunks: List[str]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "GetResponseStartEvent"


class GetResponseEndEvent(BaseEvent):
    """GetResponseEndEvent."""

    # TODO: consumes the first chunk of generators??
    # response: RESPONSE_TEXT_TYPE

    @classmethod
    def class_name(cls):
        """Class name."""
        return "GetResponseEndEvent"
