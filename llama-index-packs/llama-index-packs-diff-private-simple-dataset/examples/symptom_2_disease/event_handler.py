from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.packs.diff_private_simple_dataset.events import (
    SyntheticExampleStartEvent,
    LLMEmptyResponseEvent,
    EmptyIntersectionEvent,
)


class TooManyProblemsEncounteredError(Exception):
    pass


class DiffPrivacyEventHandler(BaseEventHandler):
    synthetic_example_starts: int = 0
    errors: int = 0
    critical_threshold: int = 250

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""
        if isinstance(event, SyntheticExampleStartEvent):
            self.synthetic_example_starts += 1
        elif isinstance(event, (LLMEmptyResponseEvent, EmptyIntersectionEvent)):
            self.errors += 1
            if self.errors > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
