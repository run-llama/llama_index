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
    num_splits: int
    t_max: int
    synthetic_example_starts: int = 0
    synthetic_example_ends: int = 0
    llm_empty_responses: int = 0
    empty_intersections: int = 0
    critical_threshold: int = 0.025  # ~2.5% error rate with OpenAI API calls

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyEventHandler"

    def compute_approximate_error_rate(self):
        """Returns an approximate error rate."""
        return (self.llm_empty_responses + self.empty_intersections) / (
            self.synthetic_example_starts * self.t_max * (self.num_splits + 1)
        )

    def handle(self, event) -> None:
        """Logic for handling event."""

        if isinstance(event, SyntheticExampleStartEvent):
            self.synthetic_example_starts += 1
        elif isinstance(event, SyntheticExampleEndEvent):
            self.synthetic_example_ends += 1
        elif isinstance(event, LLMEmptyResponseEvent):
            self.llm_empty_responses += 1

            with open("error_report.json", "w") as f:
                json.dump(self.dict(), f)

            if self.compute_approximate_error_rate() > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
        elif isinstance(event, EmptyIntersectionEvent):
            self.empty_intersections += 1

            with open("error_report.json", "w") as f:
                json.dump(self.dict(), f)

            if self.compute_approximate_error_rate() > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
