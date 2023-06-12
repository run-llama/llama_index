"""Answer inserter."""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.query_engine.flare.schema import QueryTask
from llama_index.prompts.base import Prompt
from llama_index.indices.service_context import ServiceContext


class BaseLookaheadAnswerInserter(ABC):
    """Lookahead answer inserter.

    These are responsible for insert answers into a lookahead answer template.

    E.g.
    lookahead answer: Red is for [Search(What is the meaning of Ghana's
        flag being red?)], green for forests, and gold for mineral wealth.
    query: What is the meaning of Ghana's flag being red?
    query answer: "the blood of those who died in the country's struggle
        for independence"
    final answer: Red is for the blood of those who died in the country's
        struggle for independence, green for forests, and gold for mineral wealth.

    """

    @abstractmethod
    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
    ) -> str:
        """Insert answers into response."""


DEFAULT_ANSWER_INSERT_PROMPT_TMPL = """
An existing 'lookahead response' is given below. The lookahead response
contains `[Search(query)]` tags. Some queries have been executed and the
response retrieved. The queries and answers are also given below.
Given the initial lookahead template, and also queries and answers,
please 'fill in' the lookahead template with the appropriate answers.

NOTE: the lookahead template may not be a complete sentence and may
contain trailing/leading commas, etc. Please preserve the original
formatting of the lookahead template and only replace the `[Search(query)]`
tags with the answers.

NOTE: the lookahead template may contain multiple `[Search(query)]` tags
    and only a subset of these queries have been executed.
    Do not replace the `[Search(query)]` tags that have not been executed.

Lookahead Template:
Red is for [Search(What is the meaning of Ghana's \
    flag being red?)], green for forests, and gold for mineral wealth.

Query-Answer Pairs:
Query: What is the meaning of Ghana's flag being red?
Answer: The red represents the blood of those who died in the country's struggle \
    for independence

Filled in Answers:
Red is for the blood of those who died in the country's struggle for independence, \
    green for forests, and gold for mineral wealth.

Lookahead Template:
, the city contains a population of [Search(What is the population \
    of New York City?)]

Query-Answer Pairs:
Query: What is the population of New York City?
Answer: The population of New York City is 8.4 million

Synthesized Response:
, the city contains a population of 8.4 million

Lookahead Template:
{lookahead_response}

Query-Answer Pairs:
{query_answer_pairs}

Synthesized Response:
"""
DEFAULT_ANSWER_INSERT_PROMPT = Prompt(DEFAULT_ANSWER_INSERT_PROMPT_TMPL)


class LLMLookaheadAnswerInserter(BaseLookaheadAnswerInserter):
    """LLM Lookahead answer inserter.

    Takes in a lookahead response and a list of query tasks, and the
        lookahead answers, and inserts the answers into the lookahead response.

    Args:
        service_context (ServiceContext): Service context.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        answer_insert_prompt: Optional[Prompt] = None,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._answer_insert_prompt = (
            answer_insert_prompt or DEFAULT_ANSWER_INSERT_PROMPT
        )

    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
    ) -> str:
        """Insert answers into response."""
        query_answer_pairs = ""
        for query_task, answer in zip(query_tasks, answers):
            query_answer_pairs += f"Query: {query_task.query_str}\nAnswer: {answer}\n"

        response, fmt_prompt = self._service_context.llm_predictor.predict(
            self._answer_insert_prompt,
            lookahead_response=response,
            query_answer_pairs=query_answer_pairs,
        )
        return response


class DirectLookaheadAnswerInserter(BaseLookaheadAnswerInserter):
    """Direct lookahead answer inserter.

    Simple inserter module that directly inserts answers into
        the [Search(query)] tags in the lookahead response.

    Args:
        service_context (ServiceContext): Service context.

    """

    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
    ) -> str:
        """Insert answers into response."""
        for query_task, answer in zip(query_tasks, answers):
            response = (
                response[: query_task.start_idx]
                + answer
                + response[query_task.end_idx + 1 :]
            )
        return response
