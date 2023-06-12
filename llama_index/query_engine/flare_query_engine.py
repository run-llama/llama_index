"""Query engines based on the FLARE paper.

Active Retrieval Augmented Generation.

"""

from abc import ABC, abstractmethod
from langchain.input import print_text
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.prompts.base import Prompt
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.output_parsers.base import BaseOutputParser
from llama_index.callbacks.base import CallbackManager


# These prompts are taken from the FLARE repo:
# https://github.com/jzbjyb/FLARE/blob/main/src/templates.py

DEFAULT_EXAMPLES = """
Query: But what are the risks during production of nanomaterials?
Answer: [Search(What are some nanomaterial production risks?)]

Query: The colors on the flag of Ghana have the following meanings.
Answer: Red is for [Search(What is the meaning of Ghana's flag being red?)], green for forests, and gold for mineral wealth.

Query: What did the author do during his time in college?
Answer: The author took classes in [Search(What classes did the author take in college?)].

"""

DEFAULT_FIRST_SKILL = f"""\
Skill 1. Use the Search API to look up relevant information by writing "[Search(query)]" where "query" is the search query you want to look up. For example: 
{DEFAULT_EXAMPLES}

"""

DEFAULT_SECOND_SKILL = f"""\
Skill 2. Solve more complex generation tasks by thinking step by step. For example:

Query: Give a summary of the author's life and career.
Answer: The author was born in 1990. Growing up, he [Search(What did the author do during his childhood?)].

Query: Can you write a summary of the Great Gatsby.
Answer: The Great Gatsby is a novel written by F. Scott Fitzgerald. It is about [Search(What is the Great Gatsby about?)].

"""

DEFAULT_END = """
Now given the following task, please provide the response. You may use the Search API \
"[Search(query)]" whenever possible.
If the answer is complete and no longer contains any "[Search(query)]" tags, write "done" to finish the task.
Do not write "done" if the answer still contains "[Search(query)]" tags.
Do not make up answers. It is better to use "[Search(query)]" tags than .

"""

DEFAULT_INSTRUCT_PROMPT_TMPL = (
    DEFAULT_FIRST_SKILL
    + DEFAULT_SECOND_SKILL
    + DEFAULT_END
    + (
        """
Query: {query_str}
Answer: {existing_answer}"""
    )
)

DEFAULT_INSTRUCT_PROMPT = Prompt(DEFAULT_INSTRUCT_PROMPT_TMPL)


def default_parse_is_done_fn(response: str) -> bool:
    """Default parse is done function."""
    return "done" in response.lower()


def default_format_done_answer(response: str) -> str:
    """Default format done answer."""
    return response.replace("done", "").strip()


class IsDoneOutputParser(BaseOutputParser):
    """Is done output parser."""

    def __init__(
        self,
        is_done_fn: Optional[Callable[[str], bool]] = None,
        fmt_answer_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Init params."""
        self._is_done_fn = is_done_fn or default_parse_is_done_fn
        self._fmt_answer_fn = fmt_answer_fn or default_format_done_answer

    def parse(self, output: str) -> Any:
        """Parse output."""
        is_done = default_parse_is_done_fn(output)
        if is_done:
            return True, self._fmt_answer_fn(output)
        else:
            return False, output

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError


class QueryTaskOutputParser(BaseOutputParser):
    """QueryTask output parser.

    By default, parses output that contains "[Search(query)]" tags.

    """

    def parse(self, output: str) -> Any:
        """Parse output."""
        query_tasks = []
        for idx, char in enumerate(output):
            if char == "[":
                start_idx = idx
            elif char == "]":
                end_idx = idx
                raw_query_str = output[start_idx + 1 : end_idx]
                print(raw_query_str)
                query_str = raw_query_str.split("(")[1].split(")")[0]
                query_tasks.append(QueryTask(query_str, start_idx, end_idx))
        return query_tasks

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError


@dataclass
class QueryTask:
    """Query task."""

    query_str: str
    start_idx: int
    end_idx: int


LOOKAHEAD_ANSWER_FN_TYPE = Callable[[str, List[QueryTask], List[str]], str]


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


class FLAREInstructQueryEngine(BaseQueryEngine):
    """FLARE Instruct query engine.

    This is the version of FLARE that uses retrieval-encouraging instructions.

    Args:

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        service_context: Optional[ServiceContext] = None,
        instruct_prompt: Optional[Prompt] = None,
        lookahead_answer_inserter: Optional[BaseLookaheadAnswerInserter] = None,
        done_output_parser: Optional[IsDoneOutputParser] = None,
        query_task_output_parser: Optional[QueryTaskOutputParser] = None,
        max_iterations: int = 10,
        max_lookahead_query_tasks: int = 1,
        num_prefix_chars: int = 20,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__(callback_manager=callback_manager)
        self._query_engine = query_engine
        self._service_context = service_context or ServiceContext.from_defaults()
        self._instruct_prompt = instruct_prompt or DEFAULT_INSTRUCT_PROMPT
        self._lookahead_answer_inserter = lookahead_answer_inserter or (
            LLMLookaheadAnswerInserter(service_context=self._service_context)
        )
        self._done_output_parser = done_output_parser or IsDoneOutputParser()
        self._query_task_output_parser = (
            query_task_output_parser or QueryTaskOutputParser()
        )
        self._max_iterations = max_iterations
        self._max_lookahead_query_tasks = max_lookahead_query_tasks
        self._num_prefix_chars = num_prefix_chars
        self._verbose = verbose

    def _get_relevant_lookahead_response(
        self, updated_lookahead_resp: str, num_prefix_chars: int
    ) -> str:
        """Get relevant lookahead response."""
        # if there's remaining query tasks, then truncate the response
        # until the start position of the first tag
        # there may be remaining query tasks because the _max_lookahead_query_tasks
        # is less than the total number of generated [Search(query)] tags
        remaining_query_tasks = self._query_task_output_parser.parse(
            updated_lookahead_resp
        )
        if len(remaining_query_tasks) == 0:
            relevant_lookahead_resp = updated_lookahead_resp
        else:
            first_task = remaining_query_tasks[0]
            relevant_lookahead_resp = updated_lookahead_resp[: first_task.start_idx]
        # remove prefix from lookahead resp
        relevant_lookahead_resp_wo_prefix = relevant_lookahead_resp[num_prefix_chars:]
        return relevant_lookahead_resp_wo_prefix

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query and get response."""
        print_text(f"Query: {query_bundle.query_str}\n", color="green")
        cur_response = ""
        source_nodes = []
        for iter in range(self._max_iterations):
            if self._verbose:
                print_text(f"Current response: {cur_response}\n", color="blue")
            # generate "lookahead response" that contains "[Search(query)]" tags
            # e.g.
            # The colors on the flag of Ghana have the following meanings. Red is
            # for [Search(Ghana flag meaning)],...
            lookahead_resp, fmt_response = self._service_context.llm_predictor.predict(
                self._instruct_prompt,
                query_str=query_bundle.query_str,
                existing_answer=cur_response,
            )
            lookahead_resp = lookahead_resp.strip()
            if self._verbose:
                print_text(f"Lookahead response: {lookahead_resp}\n", color="pink")

            is_done, fmt_lookahead = self._done_output_parser.parse(lookahead_resp)
            if is_done:
                cur_response = cur_response.strip() + " " + fmt_lookahead.strip()
                break

            # parse lookahead response into query tasks
            query_tasks = self._query_task_output_parser.parse(lookahead_resp)

            # get answers for each query task
            query_tasks = query_tasks[: self._max_lookahead_query_tasks]
            query_answers = []
            source_nodes = []
            for idx, query_task in enumerate(query_tasks):
                answer_obj = self._query_engine.query(query_task.query_str)
                if not isinstance(answer_obj, Response):
                    raise ValueError(
                        f"Expected Response object, got {type(answer_obj)} instead."
                    )
                query_answer = answer_obj.response
                query_answers.append(query_answer)
                source_nodes.extend(answer_obj.source_nodes)

            num_prefix_chars = min(self._num_prefix_chars, len(cur_response))
            lookahead_resp_w_prefix = cur_response[-num_prefix_chars:] + lookahead_resp

            # fill in the lookahead response template with the query answers
            # from the query engine
            updated_lookahead_resp = self._lookahead_answer_inserter.insert(
                lookahead_resp_w_prefix, query_tasks, query_answers
            )

            # get "relevant" lookahead response by truncating the updated
            # lookahead response until the start position of the first tag
            # also remove the prefix from the lookahead response, so that
            # we can concatenate it with the existing response
            relevant_lookahead_resp_wo_prefix = self._get_relevant_lookahead_response(
                updated_lookahead_resp, num_prefix_chars
            )

            if self._verbose:
                print_text(
                    f"Updated lookahead response: {relevant_lookahead_resp_wo_prefix}\n",
                    color="pink",
                )

            # append the relevant lookahead response to the final response
            cur_response = (
                cur_response.strip() + " " + relevant_lookahead_resp_wo_prefix.strip()
            )

        # NOTE: at the moment, does not support streaming
        return Response(response=cur_response)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)
