"""Query engines based on the FLARE paper.

Active Retrieval Augmented Generation.

"""

from typing import Any, Dict, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.query_engine.flare.answer_inserter import (
    BaseLookaheadAnswerInserter,
    LLMLookaheadAnswerInserter,
)
from llama_index.core.query_engine.flare.output_parser import (
    IsDoneOutputParser,
    QueryTaskOutputParser,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings, llm_from_settings_or_context
from llama_index.core.utils import print_text

# These prompts are taken from the FLARE repo:
# https://github.com/jzbjyb/FLARE/blob/main/src/templates.py

DEFAULT_EXAMPLES = """
Query: But what are the risks during production of nanomaterials?
Answer: [Search(What are some nanomaterial production risks?)]

Query: The colors on the flag of Ghana have the following meanings.
Answer: Red is for [Search(What is the meaning of Ghana's flag being red?)], \
    green for forests, and gold for mineral wealth.

Query: What did the author do during his time in college?
Answer: The author took classes in [Search(What classes did the author take in \
    college?)].

"""

DEFAULT_FIRST_SKILL = f"""\
Skill 1. Use the Search API to look up relevant information by writing \
    "[Search(query)]" where "query" is the search query you want to look up. \
    For example:
{DEFAULT_EXAMPLES}

"""

DEFAULT_SECOND_SKILL = """\
Skill 2. Solve more complex generation tasks by thinking step by step. For example:

Query: Give a summary of the author's life and career.
Answer: The author was born in 1990. Growing up, he [Search(What did the \
    author do during his childhood?)].

Query: Can you write a summary of the Great Gatsby.
Answer: The Great Gatsby is a novel written by F. Scott Fitzgerald. It is about \
    [Search(What is the Great Gatsby about?)].

"""

DEFAULT_END = """
Now given the following task, and the stub of an existing answer, generate the \
next portion of the answer. You may use the Search API \
"[Search(query)]" whenever possible.
If the answer is complete and no longer contains any "[Search(query)]" tags, write \
    "done" to finish the task.
Do not write "done" if the answer still contains "[Search(query)]" tags.
Do not make up answers. It is better to generate one "[Search(query)]" tag and stop \
generation
than to fill in the answer with made up information with no "[Search(query)]" tags
or multiple "[Search(query)]" tags that assume a structure in the answer.
Try to limit generation to one sentence if possible.

"""

DEFAULT_INSTRUCT_PROMPT_TMPL = (
    DEFAULT_FIRST_SKILL
    + DEFAULT_SECOND_SKILL
    + DEFAULT_END
    + (
        """
Query: {query_str}
Existing Answer: {existing_answer}
Answer: """
    )
)

DEFAULT_INSTRUCT_PROMPT = PromptTemplate(DEFAULT_INSTRUCT_PROMPT_TMPL)


class FLAREInstructQueryEngine(BaseQueryEngine):
    """FLARE Instruct query engine.

    This is the version of FLARE that uses retrieval-encouraging instructions.

    NOTE: this is a beta feature. Interfaces might change, and it might not
    always give correct answers.

    Args:
        query_engine (BaseQueryEngine): query engine to use
        llm (Optional[LLM]): LLM model. Defaults to None.
        service_context (Optional[ServiceContext]): service context.
            Defaults to None.
        instruct_prompt (Optional[PromptTemplate]): instruct prompt. Defaults to None.
        lookahead_answer_inserter (Optional[BaseLookaheadAnswerInserter]):
            lookahead answer inserter. Defaults to None.
        done_output_parser (Optional[IsDoneOutputParser]): done output parser.
            Defaults to None.
        query_task_output_parser (Optional[QueryTaskOutputParser]):
            query task output parser. Defaults to None.
        max_iterations (int): max iterations. Defaults to 10.
        max_lookahead_query_tasks (int): max lookahead query tasks. Defaults to 1.
        callback_manager (Optional[CallbackManager]): callback manager.
            Defaults to None.
        verbose (bool): give verbose outputs. Defaults to False.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        instruct_prompt: Optional[BasePromptTemplate] = None,
        lookahead_answer_inserter: Optional[BaseLookaheadAnswerInserter] = None,
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
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._instruct_prompt = instruct_prompt or DEFAULT_INSTRUCT_PROMPT
        self._lookahead_answer_inserter = lookahead_answer_inserter or (
            LLMLookaheadAnswerInserter(llm=self._llm)
        )
        self._done_output_parser = done_output_parser or IsDoneOutputParser()
        self._query_task_output_parser = (
            query_task_output_parser or QueryTaskOutputParser()
        )
        self._max_iterations = max_iterations
        self._max_lookahead_query_tasks = max_lookahead_query_tasks
        self._verbose = verbose

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "instruct_prompt": self._instruct_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "instruct_prompt" in prompts:
            self._instruct_prompt = prompts["instruct_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "query_engine": self._query_engine,
            "lookahead_answer_inserter": self._lookahead_answer_inserter,
        }

    def _get_relevant_lookahead_response(self, updated_lookahead_resp: str) -> str:
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
        return relevant_lookahead_resp

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
            lookahead_resp = self._llm.predict(
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
            for _, query_task in enumerate(query_tasks):
                answer_obj = self._query_engine.query(query_task.query_str)
                if not isinstance(answer_obj, Response):
                    raise ValueError(
                        f"Expected Response object, got {type(answer_obj)} instead."
                    )
                query_answer = str(answer_obj)
                query_answers.append(query_answer)
                source_nodes.extend(answer_obj.source_nodes)

            # fill in the lookahead response template with the query answers
            # from the query engine
            updated_lookahead_resp = self._lookahead_answer_inserter.insert(
                lookahead_resp, query_tasks, query_answers, prev_response=cur_response
            )

            # get "relevant" lookahead response by truncating the updated
            # lookahead response until the start position of the first tag
            # also remove the prefix from the lookahead response, so that
            # we can concatenate it with the existing response
            relevant_lookahead_resp_wo_prefix = self._get_relevant_lookahead_response(
                updated_lookahead_resp
            )

            if self._verbose:
                print_text(
                    "Updated lookahead response: "
                    + f"{relevant_lookahead_resp_wo_prefix}\n",
                    color="pink",
                )

            # append the relevant lookahead response to the final response
            cur_response = (
                cur_response.strip() + " " + relevant_lookahead_resp_wo_prefix.strip()
            )

        # NOTE: at the moment, does not support streaming
        return Response(response=cur_response, source_nodes=source_nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._query(query_bundle)
