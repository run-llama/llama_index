"""Guideline evaluation."""
import asyncio
import logging
from typing import Any, Optional, Sequence, Union, cast

from llama_index import ServiceContext
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.output_parsers import PydanticOutputParser
from llama_index.prompts import BasePromptTemplate, PromptTemplate
from llama_index.prompts.mixin import PromptDictType

logger = logging.getLogger(__name__)


DEFAULT_GUIDELINES = (
    "The response should fully answer the query.\n"
    "The response should avoid being vague or ambiguous.\n"
    "The response should be specific and use statistics or numbers when possible.\n"
)

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Here is the original query:\n"
    "Query: {query}\n"
    "Critique the following response based on the guidelines below:\n"
    "Response: {response}\n"
    "Guidelines: {guidelines}\n"
    "Now please provide constructive criticism.\n"
)


class EvaluationData(BaseModel):
    passing: bool = Field(description="Whether the response passes the guidelines.")
    feedback: str = Field(
        description="The feedback for the response based on the guidelines."
    )


class GuidelineEvaluator(BaseEvaluator):
    """Guideline evaluator.

    Evaluates whether a query and response pair passes the given guidelines.

    This evaluator only considers the query string and the response string.

    Args:
        service_context(Optional[ServiceContext]):
            The service context to use for evaluation.
        guidelines(Optional[str]): User-added guidelines to use for evaluation.
            Defaults to None, which uses the default guidelines.
        eval_template(Optional[Union[str, BasePromptTemplate]] ):
            The template to use for evaluation.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        guidelines: Optional[str] = None,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()
        self._guidelines = guidelines or DEFAULT_GUIDELINES

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._output_parser = PydanticOutputParser(output_cls=EvaluationData)
        self._eval_template.output_parser = self._output_parser

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the query and response pair passes the guidelines."""
        del contexts  # Unused
        del kwargs  # Unused
        if query is None or response is None:
            raise ValueError("query and response must be provided")

        logger.debug("prompt: %s", self._eval_template)
        logger.debug("query: %s", query)
        logger.debug("response: %s", response)
        logger.debug("guidelines: %s", self._guidelines)

        await asyncio.sleep(sleep_time_in_seconds)

        eval_response = await self._service_context.llm.apredict(
            self._eval_template,
            query=query,
            response=response,
            guidelines=self._guidelines,
        )
        eval_data = self._output_parser.parse(eval_response)
        eval_data = cast(EvaluationData, eval_data)

        return EvaluationResult(
            query=query,
            response=response,
            passing=eval_data.passing,
            score=1.0 if eval_data.passing else 0.0,
            feedback=eval_data.feedback,
        )
