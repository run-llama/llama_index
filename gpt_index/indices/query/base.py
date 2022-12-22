"""Base query classes."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, cast

from gpt_index.indices.data_structs import IndexStruct, Node
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.response_utils import give_response, refine_response
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.schema import DocumentStore
from gpt_index.utils import llm_token_counter

IS = TypeVar("IS", bound=IndexStruct)


@dataclass
class BaseQueryRunner:
    """Base query runner."""

    @abstractmethod
    def query(self, query: str, index_struct: IndexStruct) -> str:
        """Schedule a query."""
        raise NotImplementedError("Not implemented yet.")


class BaseGPTIndexQuery(Generic[IS]):
    """Base GPT Index Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    """

    def __init__(
        self,
        index_struct: IS,
        # TODO: pass from superclass
        llm_predictor: Optional[LLMPredictor] = None,
        docstore: Optional[DocumentStore] = None,
        query_runner: Optional[BaseQueryRunner] = None,
        prompt_helper: Optional[PromptHelper] = None,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        self._validate_index_struct(index_struct)
        self._index_struct = index_struct
        # create a _llm_predictor_set flag to get around mypy typing
        # hassles of keeping _llm_predictor as optional type
        if llm_predictor is None:
            self._llm_predictor_set = False
        else:
            self._llm_predictor_set = True
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._docstore = docstore
        self._query_runner = query_runner
        self._prompt_helper = prompt_helper or PromptHelper()

    def _query_node(
        self,
        query_str: str,
        node: Node,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        response: Optional[str] = None,
        verbose: bool = False,
        level: Optional[int] = None,
    ) -> str:
        """Query a given node.

        If node references a given document, then return the document.
        If node references a given index, then query the index.

        """
        level_str = "" if level is None else f"[Level {level}]"
        fmt_text_chunk = truncate_text(node.get_text(), 50)
        if verbose:
            print(f">{level_str} Searching in chunk: {fmt_text_chunk}")

        is_index_struct = False
        # if self._query_runner is not None, assume we want to do a recursive
        # query. In order to not perform a recursive query, make sure
        # _query_runner is None.
        if (
            self._query_runner is not None
            and node.ref_doc_id is not None
            and self._docstore is not None
        ):
            doc = self._docstore.get_document(node.ref_doc_id, raise_error=True)
            if isinstance(doc, IndexStruct):
                is_index_struct = True

        # If the retrieved node corresponds to another index struct, then
        # recursively query that node. Otherwise, simply return the node's text.
        if is_index_struct:
            # if is index struct, then recurse and get answer
            query_runner = cast(BaseQueryRunner, self._query_runner)
            query_response = query_runner.query(query_str, cast(IndexStruct, doc))
            if response is None:
                response = query_response
            else:
                response = refine_response(
                    self._prompt_helper,
                    self._llm_predictor,
                    response,
                    query_str,
                    query_response,
                    refine_template=refine_template,
                    verbose=verbose,
                )
        else:
            # if not index struct, then just fetch text from the node
            text = node.get_text()
            if response is None:
                response = give_response(
                    self._prompt_helper,
                    self._llm_predictor,
                    query_str,
                    text,
                    text_qa_template=text_qa_template,
                    refine_template=refine_template,
                    verbose=verbose,
                )
            else:
                response = refine_response(
                    self._prompt_helper,
                    self._llm_predictor,
                    response,
                    query_str,
                    text,
                    refine_template=refine_template,
                    verbose=verbose,
                )

        return response

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

    @abstractmethod
    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""

    @llm_token_counter("query")
    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        return self._query(query_str, verbose=verbose)

    def set_llm_predictor(self, llm_predictor: LLMPredictor) -> None:
        """Set LLM predictor."""
        self._llm_predictor = llm_predictor
        self._llm_predictor_set = True

    @property
    def llm_predictor_set(self) -> bool:
        """Get llm predictor set flag."""
        return self._llm_predictor_set

    def set_prompt_helper(self, prompt_helper: PromptHelper) -> None:
        """Set prompt helper."""
        self._prompt_helper = prompt_helper
