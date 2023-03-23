"""Base query classes."""

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar, cast

from langchain.input import print_text

from gpt_index.data_structs.data_structs import IndexStruct, Node
from gpt_index.docstore import DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.postprocessor.node import (
    BaseNodePostprocessor,
    KeywordNodePostprocessor,
    SimilarityPostprocessor,
)
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import (
    RESPONSE_TEXT_TYPE,
    ResponseBuilder,
    ResponseMode,
    TextChunk,
)
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.logger.base import LlamaLogger
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse
from gpt_index.token_counter.token_counter import llm_token_counter
from gpt_index.utils import truncate_text

IS = TypeVar("IS", bound=IndexStruct)

logger = logging.getLogger(__name__)


def _get_initial_node_postprocessors(
    required_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    similarity_cutoff: Optional[float] = None,
) -> List[BaseNodePostprocessor]:
    """Get initial node postprocessors.

    This function is to help support deprecated keyword arguments.

    """
    postprocessors: List[BaseNodePostprocessor] = []
    if required_keywords is not None or exclude_keywords is not None:
        required_keywords = required_keywords or []
        exclude_keywords = exclude_keywords or []
        keyword_postprocessor = KeywordNodePostprocessor(
            required_keywords=required_keywords, exclude_keywords=exclude_keywords
        )
        postprocessors.append(keyword_postprocessor)

    if similarity_cutoff is not None:
        similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=similarity_cutoff
        )
        postprocessors.append(similarity_postprocessor)
    return postprocessors


@dataclass
class BaseQueryRunner:
    """Base query runner."""

    @abstractmethod
    def query(self, query_bundle: QueryBundle, index_struct: IndexStruct) -> Response:
        """Schedule a query."""
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    async def aquery(
        self, query_bundle: QueryBundle, index_struct: IndexStruct
    ) -> Response:
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
        streaming (bool): Optional bool. If True, will return a StreamingResponse
            object. If False, will return a Response object.

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
        # TODO: deprecated
        required_keywords: Optional[List[str]] = None,
        # TODO: deprecated
        exclude_keywords: Optional[List[str]] = None,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        include_summary: bool = False,
        response_kwargs: Optional[Dict] = None,
        # TODO: deprecated
        similarity_cutoff: Optional[float] = None,
        use_async: bool = True,
        recursive: bool = False,
        streaming: bool = False,
        doc_ids: Optional[List[str]] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        llama_logger: Optional[LlamaLogger] = None,
        verbose: bool = False,
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

        # TODO: deprecated
        self._required_keywords = required_keywords
        # TODO: deprecated
        self._exclude_keywords = exclude_keywords
        self._response_mode = ResponseMode(response_mode)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._include_summary = include_summary

        self._response_kwargs = response_kwargs or {}
        self._use_async = use_async

        self._llama_logger = llama_logger or LlamaLogger()
        # initialize logger with metadata
        self._llama_logger.set_metadata(
            {
                "index_type": self._index_struct.get_type(),
                "doc_id": self._index_struct.get_doc_id(),
            }
        )

        self.response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            use_async=use_async,
            streaming=streaming,
            llama_logger=self._llama_logger,
        )

        # TODO: deprecated
        self.similarity_cutoff = similarity_cutoff

        self._recursive = recursive
        self._streaming = streaming
        self._doc_ids = doc_ids
        self._optimizer = optimizer

        # set default postprocessors
        init_node_preprocessors = _get_initial_node_postprocessors(
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords,
            similarity_cutoff=similarity_cutoff,
        )
        node_postprocessors = node_postprocessors or []
        self.node_preprocessors: List[BaseNodePostprocessor] = (
            init_node_preprocessors + node_postprocessors
        )
        self._verbose = verbose

        self._verbose = verbose

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
        logger.debug(f">{level_str} Searching in chunk: {fmt_text_chunk}")

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
            fmt_response = truncate_text(str(response), 200)
            if self._verbose:
                print_text(f">{level_str} Got response: {fmt_response}\n", color="blue")
            return TextChunk(str(response), is_answer=True), response
        else:
            response_txt = node.get_text()
            fmt_response = truncate_text(response_txt, 200)
            if self._verbose:
                print_text(
                    f">{level_str} Got node text: {fmt_response}\n", color="blue"
                )
            return TextChunk(response_txt), None

    async def _aget_text_from_node(
        self,
        query_bundle: QueryBundle,
        node: Node,
        level: Optional[int] = None,
    ) -> Tuple[TextChunk, Optional[Response]]:
        """Query a given node.

        If node references a given document, then return the document.
        If node references a given index, then query the index.

        """
        # TODO: refactor this to share code with _get_text_from_node
        level_str = "" if level is None else f"[Level {level}]"
        fmt_text_chunk = truncate_text(node.get_text(), 50)
        logger.debug(f">{level_str} Searching in chunk: {fmt_text_chunk}")

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
            response = await query_runner.aquery(query_bundle, cast(IndexStruct, doc))
            fmt_response = truncate_text(str(response), 200)
            if self._verbose:
                print_text(f">{level_str} Got response: {fmt_response}\n", color="blue")
            return TextChunk(str(response), is_answer=True), response
        else:
            response_txt = node.get_text()
            fmt_response = truncate_text(response_txt, 200)
            if self._verbose:
                print_text(
                    f">{level_str} Got node text: {fmt_response}\n", color="blue"
                )
            return TextChunk(response_txt), None

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

    def _give_response_for_nodes(
        self, response_builder: ResponseBuilder, query_str: str
    ) -> RESPONSE_TEXT_TYPE:
        """Give response for nodes."""
        response = response_builder.get_response(
            query_str,
            mode=self._response_mode,
            **self._response_kwargs,
        )
        return response

    async def _agive_response_for_nodes(
        self, response_builder: ResponseBuilder, query_str: str
    ) -> RESPONSE_TEXT_TYPE:
        """Give response for nodes."""
        response = await response_builder.aget_response(
            query_str,
            mode=self._response_mode,
            **self._response_kwargs,
        )
        return response

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

        postprocess_info = {"similarity_tracker": similarity_tracker}
        for node_processor in self.node_preprocessors:
            nodes = node_processor.postprocess_nodes(nodes, postprocess_info)

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

    def _get_text_tuples_from_nodes(
        self,
        query_bundle: QueryBundle,
        nodes: List[Node],
        level: Optional[int] = None,
    ) -> List[Tuple[TextChunk, Optional[Response]]]:
        """Get text tuples from nodes."""
        return [
            self._get_text_from_node(query_bundle, node, level=level) for node in nodes
        ]

    async def _aget_text_tuples_from_nodes(
        self,
        query_bundle: QueryBundle,
        nodes: List[Node],
        level: Optional[int] = None,
    ) -> List[Tuple[TextChunk, Optional[Response]]]:
        """Async get text tuples from nodes."""
        tasks = [
            self._aget_text_from_node(query_bundle, node, level=level) for node in nodes
        ]
        return await asyncio.gather(*tasks)

    def _prepare_response_builder(
        self,
        response_builder: ResponseBuilder,
        query_bundle: QueryBundle,
        node_tuples: List[Tuple[Node, Optional[float]]],
        text_tuples: List[Tuple[TextChunk, Optional[Response]]],
    ) -> None:
        """Prepare response builder and return values for query time."""
        response_builder.reset()
        for (node, similarity), (text, response) in zip(node_tuples, text_tuples):
            response_builder.add_node_as_source(node, similarity=similarity)
            if response is not None:
                # these are source nodes from within this node (when it's an index)
                for source_node in response.source_nodes:
                    response_builder.add_source_node(source_node)
            if self._optimizer is not None:
                response_builder.add_text_chunks(
                    [TextChunk(text=self._optimizer.optimize(query_bundle, text.text))]
                )
            else:
                response_builder.add_text_chunks([text])

    def _prepare_response_output(
        self,
        response_builder: ResponseBuilder,
        response_str: Optional[RESPONSE_TEXT_TYPE],
        tuples: List[Tuple[Node, Optional[float]]],
    ) -> RESPONSE_TYPE:
        """Prepare response object from response string."""
        response_extra_info = self._get_extra_info_for_response(
            [node for node, _ in tuples]
        )

        if response_str is None or isinstance(response_str, str):
            return Response(
                response_str,
                source_nodes=response_builder.get_sources(),
                extra_info=response_extra_info,
            )
        elif response_str is None or isinstance(response_str, Generator):
            return StreamingResponse(
                response_str,
                source_nodes=response_builder.get_sources(),
                extra_info=response_extra_info,
            )
        else:
            raise ValueError("Response must be a string or a generator.")

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: remove _query and just use query
        node_tuples = self.get_nodes_and_similarities_for_response(query_bundle)
        text_tuples = self._get_text_tuples_from_nodes(
            query_bundle, [t[0] for t in node_tuples]
        )

        # prepare response builder
        self._prepare_response_builder(
            self.response_builder, query_bundle, node_tuples, text_tuples
        )

        if self._response_mode != ResponseMode.NO_TEXT:
            response_str = self._give_response_for_nodes(
                self.response_builder, query_bundle.query_str
            )
        else:
            response_str = None

        return self._prepare_response_output(
            self.response_builder, response_str, node_tuples
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query asynchronously."""
        # TODO: remove _query and just use query
        node_tuples = self.get_nodes_and_similarities_for_response(query_bundle)
        text_tuples = await self._aget_text_tuples_from_nodes(
            query_bundle, [t[0] for t in node_tuples]
        )
        # define a response builder for async queries
        response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            use_async=self._use_async,
            streaming=self._streaming,
            llama_logger=self._llama_logger,
        )

        # prepare response builder
        self._prepare_response_builder(
            response_builder, query_bundle, node_tuples, text_tuples
        )

        if self._response_mode != ResponseMode.NO_TEXT:
            response_str = await self._agive_response_for_nodes(
                response_builder, query_bundle.query_str
            )
        else:
            response_str = None

        return self._prepare_response_output(
            response_builder, response_str, node_tuples
        )

    @llm_token_counter("query")
    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
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
                streaming=self._streaming,
            )
            if isinstance(response, Response):
                # NOTE: use create and refine for now (default response mode)
                response_str = response_builder.get_response(
                    query_bundle.query_str,
                    mode=self._response_mode,
                    prev_response=response.response,
                )
                response.response = cast(str, response_str)
            elif isinstance(response, StreamingResponse):
                response_gen = response_builder.get_response(
                    query_bundle.query_str,
                    mode=self._response_mode,
                    prev_response=str(response.response_gen),
                )
                response.response_gen = cast(Generator, response_gen)
            else:
                raise ValueError("Response must be a string or a generator.")

        return response

    @llm_token_counter("query")
    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = await self._aquery(query_bundle)
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
                streaming=self._streaming,
            )
            if isinstance(response, Response):
                # NOTE: use create and refine for now (default response mode)
                response_str = await response_builder.aget_response(
                    query_bundle.query_str,
                    mode=self._response_mode,
                    prev_response=response.response,
                )
                response.response = cast(str, response_str)
            elif isinstance(response, StreamingResponse):
                response_gen = await response_builder.aget_response(
                    query_bundle.query_str,
                    mode=self._response_mode,
                    prev_response=str(response.response_gen),
                )
                response.response_gen = cast(Generator, response_gen)
            else:
                raise ValueError("Response must be a string or a generator.")

        return response
