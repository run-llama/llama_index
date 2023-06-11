"""Query engines based on the FLARE paper.

Active Retrieval Augmented Generation.

"""

from langchain.input import print_text
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.prompts.base import Prompt
from llama_index.indices.base_retriever import BaseRetriever
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
Answer: The author took introductory CS classes during his freshman year. He also took some \
humanities classes, such as [Search(What humanities classes did the author take in college?)].

"""

DEFAULT_FIRST_SKILL = f"""\
Skill 1. Use the Search API to look up relevant information by writing "[Search(query)]" where "query" is the search query you want to look up. For example: 
{DEFAULT_EXAMPLES}

"""

# DEFAULT_SECOND_SKILL = f"""\
# Skill 2. Answer questions by thinking step-by-step. First, write out the reasoning steps, then draw the conclusion.

# """

# DEFAULT_THIRD_SKILL = f"""\
# Now, combine the aforementioned two skills. First, write out the reasoning steps, then draw the conclusion, \
# where the reasoning steps should also utilize the Search API "[Search(query)]" whenever possible.

# When you are done, write "done" to finish the task.\

# """

# DEFAULT_INSTRUCT_PROMPT_TMPL = (
#     DEFAULT_FIRST_SKILL
#     + DEFAULT_SECOND_SKILL
#     + DEFAULT_THIRD_SKILL
#     + (
#         """
# Query: {query_str}
# Answer: {existing_answer} \
#     """
#     )
# )

DEFAULT_END = """
Now given the query, please answer the following question. You may use the Search API \
"[Search(query)]" whenever possible.
If the answer is complete and no longer contains any "[Search(query)]" tags, write "done" to finish the task.

"""

DEFAULT_INSTRUCT_PROMPT_TMPL = (
    DEFAULT_FIRST_SKILL
    + DEFAULT_END
    + (
        """
Query: {query_str}
Answer: {existing_answer} \
    """
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
        done_output_parser: Optional[IsDoneOutputParser] = None,
        query_task_output_parser: Optional[QueryTaskOutputParser] = None,
        max_iterations: int = 10,
        max_lookahead_query_tasks: int = 1,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__(callback_manager=callback_manager)
        self._query_engine = query_engine
        self._service_context = service_context or ServiceContext.from_defaults()
        self._instruct_prompt = instruct_prompt or DEFAULT_INSTRUCT_PROMPT
        self._done_output_parser = done_output_parser or IsDoneOutputParser()
        self._query_task_output_parser = (
            query_task_output_parser or QueryTaskOutputParser()
        )
        self._max_iterations = max_iterations
        self._max_lookahead_query_tasks = max_lookahead_query_tasks
        self._verbose = verbose

    def _insert_answer(
        self,
        response: str,
        query_task: QueryTask,
        answer: str,
    ) -> str:
        """Insert answer into response."""
        return (
            response[: query_task.start_idx]
            + answer
            + response[query_task.end_idx + 1 :]
        )

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
            print(fmt_response)
            print_text(f"Lookahead response: {lookahead_resp}\n", color="pink")

            is_done, fmt_lookahead = self._done_output_parser.parse(lookahead_resp)
            if is_done:
                cur_response = cur_response.strip() + " " + fmt_lookahead.strip()
                break

            # parse lookahead response into query tasks
            query_tasks = self._query_task_output_parser.parse(lookahead_resp)

            print(f"query tasks: {query_tasks}")

            # get answers for each query task
            query_tasks = query_tasks[: self._max_lookahead_query_tasks]
            query_answers = []
            for idx, query_task in enumerate(query_tasks):
                answer_obj = self._query_engine.query(query_task.query_str)
                if not isinstance(answer_obj, Response):
                    raise ValueError(
                        f"Expected Response object, got {type(answer_obj)} instead."
                    )
                query_answer = answer_obj.response
                query_answers.append(query_answer)

            # update lookahead response
            # TODO:

            # for idx, query_task in enumerate(query_tasks):
            #     if idx >= self._max_lookahead_query_tasks:
            #         break
            #     # get answer
            #     answer_obj = self._query_engine.query(query_task.query_str)
            #     if not isinstance(answer_obj, Response):
            #         raise ValueError(
            #             f"Expected Response object, got {type(answer_obj)} instead."
            #         )
            #     query_answer = answer_obj.response
            #     print("query_answer: ", query_answer)
            #     source_nodes.extend(answer_obj.source_nodes)
            #     # replace answer
            #     lookahead_resp = self._insert_answer(
            #         lookahead_resp, query_task, query_answer
            #     )
            #     print(f"updated lookahead_resp: {lookahead_resp}")

            # get relevant lookahead response by truncation
            truncate_idx = query_tasks[-1].end_idx
            relevant_lookahead_resp = lookahead_resp[:truncate_idx]
            print("relevant_lookahead_resp: ", relevant_lookahead_resp)

            # append the relevant lookahead response to the final response
            cur_response = cur_response.strip() + " " + relevant_lookahead_resp.strip()

        # NOTE: at the moment, does not support streaming
        return Response(response=cur_response)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)
