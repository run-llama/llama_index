"""Evaluating the responses from an index."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from llama_index.response.schema import Response


@dataclass
class Evaluation:
    query: str  # The query
    response: Response  # The response
    passing: bool = False  # True if the response is correct, False otherwise
    feedback: str = ""  # Feedback for the response
    score: Optional[float] = None  # Score for the response


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate_response(
        self, query: str, response: Response, **kwargs: Any
    ) -> Evaluation:
        """Evaluate the response for a query and return an Evaluation."""
        raise NotImplementedError

    def evaluate_string(self, query: str, response: str, **kwargs: Any) -> Evaluation:
        """Evaluate the response for a query and return an Evaluation."""
        raise NotImplementedError
