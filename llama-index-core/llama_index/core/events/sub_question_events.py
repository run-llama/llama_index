"""Events for Sub Questions."""

from dataclasses import dataclass

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType
from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionAnswerPair,
)


@dataclass
class SubQuestionStartEventPayload:
    """Payload for SubQuestionStartEvent."""

    sub_question: SubQuestionAnswerPair


class SubQuestionStartEvent(CBEvent):
    """Event to indicate a tree has started."""

    sub_question: SubQuestionAnswerPair

    def __init__(self, sub_question: SubQuestionAnswerPair):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.TREE)
        self.sub_question = sub_question

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> SubQuestionStartEventPayload:
        """Return the payload for the event."""
        return SubQuestionStartEventPayload(sub_question=self.sub_question)


@dataclass
class SubQuestionEndEventPayload:
    """Payload for SubQuestionEndEvent."""

    sub_question: SubQuestionAnswerPair


class SubQuestionEndEvent(CBEvent):
    """Event to indicate a tree has ended."""

    sub_question: SubQuestionAnswerPair

    def __init__(self, sub_question: SubQuestionAnswerPair):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.TREE)
        self.sub_question = sub_question

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> SubQuestionEndEventPayload:
        """Return the payload for the event."""
        return SubQuestionEndEventPayload(sub_question=self.sub_question)
