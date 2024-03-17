from typing import Dict
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.packs.diff_private_simple_dataset.events import (
    SyntheticExampleStartEvent,
    LLMEmptyResponseEvent,
    EmptyIntersectionEvent,
)
import json


class TooManyProblemsEncounteredError(Exception):
    pass


class DiffPrivacyEventHandler(BaseEventHandler):
    synthetic_example_starts: int = 0
    errors: int = 0
    critical_threshold: int = 250
    error_counts: Dict = {
        "LLMEmptyResponseEvent": 0,
        "EmptyIntersectionEvent": 0,
    }

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
            self.error_counts[event.class_name()] = (
                self.error_counts[event.class_name()] + 1
            )
            error_report = {
                "synthetic_example_starts": self.synthetic_example_starts,
                "error_counts": self.error_counts,
            }
            with open("error_report.json", "w") as f:
                json.dump(error_report, f)
            if self.errors > self.critical_threshold:
                raise TooManyProblemsEncounteredError(
                    "There were too many errors encountered."
                )
