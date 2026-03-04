"""Test EvalQueryEngine tool."""

from typing import Optional, Sequence, Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from llama_index.core.evaluation import EvaluationResult
from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.response import Response
from llama_index.core.tools.eval_query_engine import EvalQueryEngineTool
from llama_index.core.tools.types import ToolOutput


class MockEvaluator(BaseEvaluator):
    """Mock Evaluator for testing purposes."""

    def _get_prompts(self) -> PromptDictType: ...

    def _update_prompts(self, prompts_dict: PromptDictType) -> None: ...

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...


class MockQueryEngine(CustomQueryEngine):
    """Custom query engine."""

    def custom_query(self, query_str: str) -> str:
        """Query."""
        return "custom_" + query_str


class TestEvalQueryEngineTool(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_evaluator = MockEvaluator()
        self.mock_evaluator.aevaluate = AsyncMock()
        self.mock_evaluator.aevaluate.return_value = EvaluationResult(passing=True)

        tool_name = "nice_tool"
        self.tool_input = "hello world"
        self.expected_content = f"custom_{self.tool_input}"
        self.expected_tool_output = ToolOutput(
            content=self.expected_content,
            raw_input={"input": self.tool_input},
            raw_output=Response(
                response=self.expected_content,
                source_nodes=[],
            ),
            tool_name=tool_name,
        )
        self.eval_query_engine_tool = EvalQueryEngineTool.from_defaults(
            MockQueryEngine(), evaluator=self.mock_evaluator, name=tool_name
        )

    def test_eval_query_engine_tool_with_eval_passing(self) -> None:
        """Test eval query engine tool with evaluation passing."""
        tool_output = self.eval_query_engine_tool(self.tool_input)
        self.assertEqual(self.expected_tool_output, tool_output)

    def test_eval_query_engine_tool_with_eval_failing(self) -> None:
        """Test eval query engine tool with evaluation failing."""
        evaluation_feedback = "The context does not provide a relevant answer."
        self.mock_evaluator.aevaluate.return_value = EvaluationResult(
            passing=False, feedback=evaluation_feedback
        )
        self.expected_tool_output.content = (
            "Could not use tool nice_tool because it failed evaluation.\n"
            f"Reason: {evaluation_feedback}"
        )

        tool_output = self.eval_query_engine_tool(self.tool_input)
        self.assertEqual(self.expected_tool_output, tool_output)
