from typing import Dict
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.packs.diff_private_simple_dataset.events import (
    SyntheticExampleStartEvent,
    SyntheticExampleEndEvent,
    LLMEmptyResponseEvent,
    EmptyIntersectionEvent,
)
import json


class TooManyProblemsEncounteredError(Exception):
    pass


class DiffPrivacyEventHandler(BaseEventHandler):
    synthetic_example_starts: int = 0
    synthetic_example_ends: int = 0
    llm_empty_responses: int = 0
    empty_intersections: int = 0
    critical_threshold: int = 2_500  # ~2.5% error rate with OpenAI API calls

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""

        if isinstance(event, SyntheticExampleStartEvent):
            self.synthetic_example_starts += 1
        elif isinstance(event, LLMEmptyResponseEvent):
            self.llm_empty_responses += 1

            with open("error_report.json", "w") as f:
                json.dump(self.dict(), f)

            if (
                self.llm_empty_responses + self.empty_intersections
            ) > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
        elif isinstance(event, EmptyIntersectionEvent):
            self.empty_intersections += 1

            with open("error_report.json", "w") as f:
                json.dump(self.dict(), f)

            if (
                self.llm_empty_responses + self.empty_intersections
            ) > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
