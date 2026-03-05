"""Agent goal success evaluation."""

import asyncio
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.evaluation.eval_utils import default_parser
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings

DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for AI agents.

You are given the following information:
- a user goal or task description
- the agent's final response
- optionally, a history of tool calls and their outputs
- optionally, an expected outcome for reference

Your job is to judge whether the agent successfully achieved the user's goal.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the agent completely failed to address the goal, \
you should give a score of 1.
- If the agent partially addressed the goal but with significant errors \
or missing steps, you should give a score between 2 and 3.
- If the agent successfully achieved the goal with minor issues, \
you should give a score of 4.
- If the agent perfectly achieved the goal, \
you should give a score of 5.

Example Response:
4.0
The agent correctly identified the relevant tools and produced an accurate \
    response, but included some unnecessary intermediate steps.

"""

DEFAULT_USER_TEMPLATE = """
## User Goal
{query}

## Expected Outcome
{reference_answer}

## Tool Call History
{tool_history}

## Agent Response
{generated_answer}
"""

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class AgentGoalSuccessEvaluator(BaseEvaluator):
    """
    Evaluate whether an agent successfully achieved a given goal.

    Uses an LLM judge to score the agent's response on a 1-5 scale
    based on goal achievement, considering the tool call history
    and an optional expected outcome.

    This evaluator follows the same pattern as CorrectnessEvaluator.

    Args:
        llm: The LLM to use for evaluation. Defaults to Settings.llm.
        eval_template: Custom evaluation prompt template.
        score_threshold: Minimum score for passing. Defaults to 4.0.
        parser_function: Function to parse LLM response into (score, reasoning).
            Defaults to default_parser.

    Example:
        .. code-block:: python

            from llama_index.core.evaluation import AgentGoalSuccessEvaluator

            evaluator = AgentGoalSuccessEvaluator()
            result = evaluator.evaluate(
                query="Find the weather in San Francisco",
                response="The weather in San Francisco is 65F and sunny.",
                contexts=["Called weather_api(city='San Francisco') -> '65F, sunny'"],
            )
            print(result.score)    # e.g. 5.0
            print(result.passing)  # True

    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        eval_template: Optional[Union[BasePromptTemplate, str]] = None,
        score_threshold: float = 4.0,
        parser_function: Callable[
            [str], Tuple[Optional[float], Optional[str]]
        ] = default_parser,
    ) -> None:
        self._llm = llm or Settings.llm
        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE
        self._score_threshold = score_threshold
        self.parser_function = parser_function

    def _get_prompts(self) -> PromptDictType:
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("query and response must be provided")

        tool_history = "\n".join(contexts) if contexts else "(NO TOOL HISTORY PROVIDED)"

        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference or "(NO EXPECTED OUTCOME PROVIDED)",
            tool_history=tool_history,
        )

        score, reasoning = self.parser_function(eval_response)

        return EvaluationResult(
            query=query,
            response=response,
            passing=score >= self._score_threshold if score is not None else None,
            score=score,
            feedback=reasoning,
        )
