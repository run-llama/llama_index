"""Evaluating the responses from an index."""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llama_index.response.schema import Response


@dataclass
class EvaluationResult:
    query: str  # The query
    response: Response  # The response
    passing: bool = False  # True if the response is correct, False otherwise
    feedback: str = ""  # Feedback for the response
    score: Optional[float] = None  # Score for the response


class BaseEvaluator(ABC):
    def evaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string and generated Response object."""
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [node.get_content() for node in response.source_nodes]

        return self.evaluate(
            query=query, contexts=contexts, response=response_str, **kwargs
        )

    def evaluate(
        self,
        query: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        response: Optional[str] = None,
        **kwargs: Any,
    ) -> Evaluation:
        """Run evaluation with query string, retrieved contexts, 
        and generated response string.
        """
        raise NotImplementedError


# legacy: backward compatibility
Evaluation = EvaluationResult
