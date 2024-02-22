"""Events for Reranking."""

from dataclasses import dataclass
from typing import List

from deprecated import deprecated

from llama_index.core.callbacks.schema import CBEvent, CBEventType
from llama_index.core.schema import NodeWithScore


@dataclass
class RerankingStartEventPayload:
    """Payload for RerankingStartEvent."""

    nodes: List[NodeWithScore]
    model_name: str
    query_str: str
    top_k: int


class RerankingStartEvent(CBEvent):
    """Event to indicate reranking has started."""

    nodes: List[NodeWithScore]
    model_name: str
    query_str: str
    top_k: int

    def __init__(
        self, nodes: List[NodeWithScore], model_name: str, query_str: str, top_k: int
    ):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RERANKING)
        self.nodes = nodes
        self.model_name = model_name
        self.query_str = query_str
        self.top_k = top_k

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> RerankingStartEventPayload:
        """Return the payload for the event."""
        return RerankingStartEventPayload(
            nodes=self.nodes,
            model_name=self.model_name,
            query_str=self.query_str,
            top_k=self.top_k,
        )


@dataclass
class RerankingEndEventPayload:
    """Payload for RerankingEndEvent."""

    nodes: List[NodeWithScore]


class RerankingEndEvent(CBEvent):
    """Event to indicate reranking has ended."""

    nodes: List[NodeWithScore]

    def __init__(self, nodes: List[NodeWithScore]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RERANKING)
        self.nodes = nodes

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> RerankingEndEventPayload:
        """Return the payload for the event."""
        return RerankingEndEventPayload(nodes=self.nodes)
