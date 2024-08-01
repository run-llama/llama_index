from typing import Any, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    BaseEvaluator,
    EvaluationResult,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput


DEFAULT_NAME = "query_engine_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and get back a natural language response.
"""
FAILED_TOOL_OUTPUT_TEMPLATE = (
    "Could not use tool {tool_name} because it failed evaluation.\n" "Reason: {reason}"
)


class EvalQueryEngineTool(QueryEngineTool):
    """Evaluating query engine tool.

    A tool that makes use of a query engine and an evaluator, where the
    evaluation of the query engine response will determine the tool output.

    Args:
        evaluator (BaseEvaluator): A query engine.
        query_engine (BaseQueryEngine): A query engine.
        metadata (ToolMetadata): The associated metadata of the query engine.
    """

    _evaluator: BaseEvaluator
    _failed_tool_output_template: str

    def __init__(
        self,
        evaluator: BaseEvaluator,
        *args,
        failed_tool_output_template: str = FAILED_TOOL_OUTPUT_TEMPLATE,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._evaluator = evaluator
        self._failed_tool_output_template = failed_tool_output_template

    def _process_tool_output(
        self,
        tool_output: ToolOutput,
        evaluation_result: EvaluationResult,
    ) -> ToolOutput:
        if evaluation_result.passing:
            return tool_output

        tool_output.content = self._failed_tool_output_template.format(
            tool_name=self.metadata.name,
            reason=evaluation_result.feedback,
        )
        return tool_output

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        evaluator: Optional[BaseEvaluator] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        resolve_input_errors: bool = True,
    ) -> "EvalQueryEngineTool":
        return cls(
            evaluator=evaluator or AnswerRelevancyEvaluator(),
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=name or DEFAULT_NAME,
                description=description or DEFAULT_DESCRIPTION,
            ),
            resolve_input_errors=resolve_input_errors,
        )

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        tool_output = super().call(*args, **kwargs)
        evaluation_results = self._evaluator.evaluate_response(
            tool_output.raw_input["input"], tool_output.raw_output
        )
        return self._process_tool_output(tool_output, evaluation_results)

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        tool_output = await super().acall(*args, **kwargs)
        evaluation_results = await self._evaluator.aevaluate_response(
            tool_output.raw_input["input"], tool_output.raw_output
        )
        return self._process_tool_output(tool_output, evaluation_results)
