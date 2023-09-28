"""Pairwise evaluation."""

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index import ServiceContext
from typing import Optional, Union, Sequence, Any
from llama_index.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)


DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, 
- a reference answer, and
- a candidate answer.

Your job is to output whether the candidate answer is better than the reference \
answer in answering the user query. \
'YES' means the candidate answer is better, 'NO' means the \
reference answer is better, and 'TIE' means they are equally good.

Please output two lines: 
- first line: the word YES, NO, or TIE
- second line: a short explanation for your decision.
"""

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Candidate Answer
{generated_answer}
"""

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class PairwiseComparisonEvaluator(BaseEvaluator):
    """Pairwise comparison evaluator.

    Evaluates the quality of a response vs. a "reference" response given a question by
    having an LLM judge which response is better.

    Outputs whether the `response` given is better than the `reference` response.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        eval_template: Optional[Union[BasePromptTemplate, str]] = None,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        if query is None or response is None or reference is None:
            print(query, response, reference, flush=True)
            raise ValueError("query, response, and reference must be provided")

        eval_response = await self._service_context.llm_predictor.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
        )

        eval_decision, eval_reason = eval_response.split("\n")

        # Extract from response
        if "yes" in eval_decision.lower():
            passing: Optional[bool] = True
            score = 1.0
        elif "no" in eval_decision.lower():
            passing = False
            score = 0.0
        else:
            passing = None
            score = 0.5

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=eval_reason.strip(),
        )
