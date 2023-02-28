"""Base query classes."""

import logging
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, cast

from gpt_index.data_structs.data_structs import IndexStruct, Node
from gpt_index.docstore import DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder, ResponseMode, TextChunk
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import Response
from gpt_index.token_counter.token_counter import llm_token_counter
from gpt_index.utils import truncate_text

IS = TypeVar("IS", bound=IndexStruct)


@dataclass
class BaseQueryRunner:
    """Base query runner."""

    @abstractmethod
    def query(self, query_bundle: QueryBundle, index_struct: IndexStruct) -> Response:
        """Schedule a query."""
        raise NotImplementedError("Not implemented yet.")


class BaseGPTIndexQuery(Generic[IS]):
    """Base LlamaIndex Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    Args:
        llm_predictor (LLMPredictor): Optional LLMPredictor object. If not provided,
            will use the default LLMPredictor (text-davinci-003)
        prompt_helper (PromptHelper): Optional PromptHelper object. If not provided,
            will use the default PromptHelper.
        required_keywords (List[str]): Optional list of keywords that must be present
            in nodes. Can be used to query most indices (tree index is an exception).
        exclude_keywords (List[str]): Optional list of keywords that must not be
            present in nodes. Can be used to query most indices (tree index is an
            exception).
        response_mode (ResponseMode): Optional ResponseMode. If not provided, will
            use the default ResponseMode.
        text_qa_template (QuestionAnswerPrompt): Optional QuestionAnswerPrompt object.
            If not provided, will use the default QuestionAnswerPrompt.
        refine_template (RefinePrompt): Optional RefinePrompt object. If not provided,
            will use the default RefinePrompt.
        include_summary (bool): Optional bool. If True, will also use the summary
            text of the index when generating a response (the summary text can be set
            through `index.set_text("<text>")`).
        similarity_cutoff (float): Optional float. If set, will filter out nodes with
            similarity below this cutoff threshold when computing the response

    """

    def __init__(
        self,
        index_struct: IS,
        # TODO: pass from superclass
        llm_predictor: Optional[LLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        docstore: Optional[DocumentStore] = None,
        query_runner: Optional[BaseQueryRunner] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        include_summary: bool = False,
        response_kwargs: Optional[Dict] = None,
        similarity_cutoff: Optional[float] = None,
        use_async: bool = True,
        recursive: bool = False,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        self._validate_index_struct(index_struct)
        self._index_struct = index_struct
        self._llm_predictor = llm_predictor or LLMPredictor()
        # NOTE: the embed_model isn't used in all indices
        self._embed_model = embed_model or OpenAIEmbedding()
        self._docstore = docstore
        self._query_runner = query_runner
        # TODO: make this a required param
        if prompt_helper is None:
            raise ValueError("prompt_helper must be provided.")
        self._prompt_helper = cast(PromptHelper, prompt_helper)

        self._required_keywords = required_keywords
        self._exclude_keywords = exclude_keywords
        self._response_mode = ResponseMode(response_mode)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT
        self._include_summary = include_summary

        self._response_kwargs = response_kwargs or {}
        self._use_async = use_async
        self.response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            use_async=use_async,
        )

        self.similarity_cutoff = similarity_cutoff
        self._recursive = recursive

    def _should_use_node(
        self, node: Node, similarity_tracker: Optional[SimilarityTracker] = None
    ) -> bool:
        """Run node through filters to determine if it should be used."""
        words = re.findall(r"\w+", node.get_text())
        if self._required_keywords is not None:
            for w in self._required_keywords:
                if w not in words:
                    return False

        if self._exclude_keywords is not None:
            for w in self._exclude_keywords:
                if w in words:
                    return False

        sim_cutoff_exists = (
            similarity_tracker is not None and self.similarity_cutoff is not None
        )

        if sim_cutoff_exists:
            similarity = cast(SimilarityTracker, similarity_tracker).find(node)
            if similarity is None:
                return False
            if cast(float, similarity) < cast(float, self.similarity_cutoff):
                return False

        return True

    def _get_text_from_node(
        self,
        query_bundle: QueryBundle,
        node: Node,
        level: Optional[int] = None,
    ) -> Tuple[TextChunk, Optional[Response]]:
        """Query a given node.

        If node references a given document, then return the document.
        If node references a given index, then query the index.

        """
        level_str = "" if level is None else f"[Level {level}]"
        fmt_text_chunk = truncate_text(node.get_text(), 50)
        logging.debug(f">{level_str} Searching in chunk: {fmt_text_chunk}")

        is_index_struct = False
        # if recursive and self._query_runner is not None,
        # assume we want to do a recursive
        # query. In order to not perform a recursive query, make sure
        # _query_runner is None.
        if (
            self._recursive
            and self._query_runner is not None
            and node.ref_doc_id is not None
            and self._docstore is not None
        ):
            doc = self._docstore.get_document(node.ref_doc_id, raise_error=False)
            # NOTE: old version of the docstore contain both documents and index_struct,
            # whereas new versions of the docstore only contain the index struct
            if doc is not None and isinstance(doc, IndexStruct):
                is_index_struct = True

        if is_index_struct:
            query_runner = cast(BaseQueryRunner, self._query_runner)
            response = query_runner.query(query_bundle, cast(IndexStruct, doc))
            return TextChunk(str(response), is_answer=True), response
        else:
            text = node.get_text()
            return TextChunk(text), None

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

    def _give_response_for_nodes(self, query_str: str) -> str:
        """Give response for nodes."""
        response = self.response_builder.get_response(
            query_str,
            mode=self._response_mode,
            **self._response_kwargs,
        )

        return response or ""

    def get_nodes_and_similarities_for_response(
        self, query_bundle: QueryBundle
    ) -> List[Tuple[Node, Optional[float]]]:
        """Get list of tuples of node and similarity for response.

        First part of the tuple is the node.
        Second part of tuple is the distance from query to the node.
        If not applicable, it's None.
        """
        similarity_tracker = SimilarityTracker()
        nodes = self._get_nodes_for_response(
            query_bundle, similarity_tracker=similarity_tracker
        )
        nodes = [
            node for node in nodes if self._should_use_node(node, similarity_tracker)
        ]

        # TODO: create a `display` method to allow subclasses to print the Node
        return similarity_tracker.get_zipped_nodes(nodes)

    @abstractmethod
    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""

    def _get_extra_info_for_response(
        self,
        nodes: List[Node],
    ) -> Optional[Dict[str, Any]]:
        """Get extra info for response."""
        return None

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        self.response_builder.reset()
        # TODO: remove _query and just use query
        tuples = self.get_nodes_and_similarities_for_response(query_bundle)

        for node, similarity in tuples:
            text, response = self._get_text_from_node(query_bundle, node)
            self.response_builder.add_node_as_source(node, similarity=similarity)
            if response is not None:
                # these are source nodes from within this node (when it's an index)
                for source_node in response.source_nodes:
                    self.response_builder.add_source_node(source_node)
            self.response_builder.add_text_chunks([text])

        if self._response_mode != ResponseMode.NO_TEXT:
            response_str = self._give_response_for_nodes(query_bundle.query_str)
        else:
            response_str = None

        response_extra_info = self._get_extra_info_for_response(
            [node for node, _ in tuples]
        )

        return Response(
            response_str,
            source_nodes=self.response_builder.get_sources(),
            extra_info=response_extra_info,
        )

    @llm_token_counter("query")
    def query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        response = self._query(query_bundle)
        # if include_summary is True, then include summary text in answer
        # summary text is set through `set_text` on the underlying index.
        # TODO: refactor response builder to be in the __init__
        if self._response_mode != ResponseMode.NO_TEXT and self._include_summary:
            response_builder = ResponseBuilder(
                self._prompt_helper,
                self._llm_predictor,
                self.text_qa_template,
                self.refine_template,
                texts=[TextChunk(self._index_struct.get_text())],
            )
            # NOTE: use create and refine for now (default response mode)
            response.response = response_builder.get_response(
                query_bundle.query_str,
                mode=self._response_mode,
                prev_response=response.response,
            )

        return response
