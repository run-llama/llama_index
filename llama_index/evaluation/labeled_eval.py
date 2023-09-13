from typing import Optional

from llama_index.evaluation.base import BaseEvaluator, Evaluation
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
)
from llama_index.response.schema import Response

DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, 
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.
"""

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class LabeledEvaluator(BaseEvaluator):
    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        eval_template: Optional[BasePromptTemplate] = None,
        score_threshold: float = 4.0,
        llm: Optional[str] = None,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()
        self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE
        self._score_threshold = score_threshold

    def evaluate_string(self, query: str, response: str, reference: str) -> Evaluation:
        eval_response = self._service_context.llm_predictor.predict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
        )

        # Extract from response
        score_str, reasoning_str = eval_response.split("\n", 1)
        score = float(score_str)
        reasoning = reasoning_str.lstrip("\n")

        return Evaluation(
            query=query,
            response=response,
            passing=score >= self._score_threshold,
            score=score,
            feedback=reasoning,
        )

    def evaluate_response(
        self, query: str, response: Response, reference: str
    ) -> Evaluation:
        return self.evaluate_string(query, response.response, reference)
