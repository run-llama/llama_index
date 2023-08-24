import logging
from typing import Optional

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from llama_index.bridge.langchain import PydanticOutputParser
from llama_index.evaluation.base import BaseEvaluator, Evaluation
from llama_index.indices.base import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.response.schema import Response

logger = logging.getLogger(__name__)


class GuidelineEvaluator(BaseEvaluator):
    """An evaluator which uses guidelines to evaluate a response.

    Args:
        service_context(ServiceContext): The service context to use for evaluation.
        guidelines(str): User-added guidelines to use for evaluation.
        eval_template(str): The template to use for evaluation.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        guidelines: Optional[str] = None,
        eval_template: Optional[str] = None,
    ) -> None:
        self.service_context = service_context or ServiceContext.from_defaults()
        self.guidelines = guidelines or DEFAULT_GUIDELINES
        self.eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

    def evaluate_response(self, query: str, response: Response) -> Evaluation:
        """Evaluate the response for a query and an Evaluation."""
        parser: PydanticOutputParser[EvaluationData] = PydanticOutputParser(
            pydantic_object=EvaluationData
        )
        format_instructions = parser.get_format_instructions()
        response_str = response.response
        prompt = Prompt(self.eval_template)
        logger.debug("prompt: %s", prompt)
        logger.debug("query: %s", query)
        logger.debug("response: %s", response_str)
        logger.debug("guidelines: %s", self.guidelines)
        logger.debug("format_instructions: %s", format_instructions)
        eval_response = self.service_context.llm_predictor.predict(
            prompt,
            query=query,
            response=response_str,
            guidelines=self.guidelines,
            format_instructions=format_instructions,
        )
        eval_data = parser.parse(eval_response)
        return Evaluation(query, response, eval_data.passing, eval_data.feedback)


DEFAULT_GUIDELINES = (
    "The response should fully answer the query.\n"
    "The response should avoid being vague or ambiguous.\n"
    "The response should be specific and use statistics or numbers when possible.\n"
)

DEFAULT_EVAL_TEMPLATE = (
    "Here is the original query:\n"
    "Query: {query}\n"
    "Critique the following response based on the guidelines below:\n"
    "Response: {response}\n"
    "Guidelines: {guidelines}\n"
    "Now please provide constructive criticism in the following format:\n"
    "{format_instructions}"
)


class EvaluationData(BaseModel):
    passing: bool = Field(description="Whether the response passes the guidelines.")
    feedback: str = Field(description="The feedback for the response based on the guidelines.")
