"""Tool call correctness evaluation for agents."""

from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.evaluation.agent.utils import compare_tool_calls
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType


class ToolCallCorrectnessEvaluator(BaseEvaluator):
    """
    Evaluate whether an agent made the correct tool calls.

    Compares a list of expected tool calls against actual tool calls,
    checking tool names and keyword arguments. This is a deterministic
    evaluator that does not use an LLM.

    Each tool call is a dict with keys:
        - tool_name (str): Name of the tool.
        - tool_kwargs (dict): Keyword arguments passed to the tool.

    Args:
        ordered: If True, tool calls must appear in the same order.
            Defaults to False (unordered matching).
        strict_kwargs: If True, kwargs must match exactly.
            If False, expected kwargs must be a subset of actual kwargs.
            Defaults to False.
        threshold: Minimum score (fraction of matched calls) for passing.
            Defaults to 1.0 (all expected calls must match).

    Example:
        .. code-block:: python

            from llama_index.core.evaluation import ToolCallCorrectnessEvaluator

            evaluator = ToolCallCorrectnessEvaluator()
            result = evaluator.evaluate(
                expected_tool_calls=[
                    {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
                ],
                actual_tool_calls=[
                    {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
                ],
            )
            print(result.score)  # 1.0

    """

    def __init__(
        self,
        ordered: bool = False,
        strict_kwargs: bool = False,
        threshold: float = 1.0,
    ) -> None:
        self._ordered = ordered
        self._strict_kwargs = strict_kwargs
        self._threshold = threshold

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        expected_tool_calls: Optional[List[Dict[str, Any]]] = None,
        actual_tool_calls: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del query, response, contexts, reference  # Unused

        if expected_tool_calls is None or actual_tool_calls is None:
            raise ValueError(
                "Both expected_tool_calls and actual_tool_calls must be provided"
            )

        comparison = compare_tool_calls(
            expected=expected_tool_calls,
            actual=actual_tool_calls,
            ordered=self._ordered,
            strict_kwargs=self._strict_kwargs,
        )

        score = comparison.score
        passing = score >= self._threshold

        feedback_parts = [f"Matched {comparison.matched}/{comparison.total_expected}"]
        if comparison.unmatched_expected:
            names = [tc.get("tool_name", "?") for tc in comparison.unmatched_expected]
            feedback_parts.append(f"Missing expected: {names}")
        if comparison.unmatched_actual:
            names = [tc.get("tool_name", "?") for tc in comparison.unmatched_actual]
            feedback_parts.append(f"Unexpected calls: {names}")

        return EvaluationResult(
            score=score,
            passing=passing,
            feedback=". ".join(feedback_parts),
        )
