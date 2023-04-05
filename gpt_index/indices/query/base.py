"""Base query classes."""

import logging
from abc import ABC
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from langchain.input import print_text

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.docstore import DocumentStore
from gpt_index.indices.postprocessor.node import (
    BaseNodePostprocessor,
    KeywordNodePostprocessor,
    SimilarityPostprocessor,
)
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import (
    ResponseBuilder,
    ResponseMode,
    TextChunk,
)
from gpt_index.indices.service_context import ServiceContext
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import (
    RESPONSE_TYPE,
    Response,
    StreamingResponse,
)
from gpt_index.token_counter.token_counter import llm_token_counter
from gpt_index.types import RESPONSE_TEXT_TYPE
from gpt_index.utils import truncate_text

# to prevent us from having to remove all instances of v2 later
IndexStruct = V2IndexStruct
IS = TypeVar("IS", bound=V2IndexStruct)

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


class BaseGPTIndexQuery(Generic[IS], ABC):
    """Base LlamaIndex Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    Args:
        service_context (ServiceContext): service context container (contains components
            like LLMPredictor, PromptHelper).
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
        service_context: ServiceContext,
        docstore: Optional[DocumentStore] = None,
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
        use_async: bool = False,
        streaming: bool = False,
        doc_ids: Optional[List[str]] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        self._validate_index_struct(index_struct)
        self._index_struct = index_struct
        if docstore is None:
            raise ValueError("docstore must be provided.")
        self._docstore = docstore
        self._service_context = service_context

        self._response_mode = ResponseMode(response_mode)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._include_summary = include_summary

        self._response_kwargs = response_kwargs or {}
        self._use_async = use_async

        # initialize logger with metadata
        if self._service_context.llama_logger is not None:
            self._service_context.llama_logger.set_metadata(
                {
                    "index_type": self._index_struct.get_type(),
                    "index_id": self._index_struct.index_id,
                }
            )

        self.response_builder = ResponseBuilder(
            self._service_context,
            self.text_qa_template,
            self.refine_template,
            use_async=use_async,
            streaming=streaming,
        )

        # TODO: deprecated
        self.similarity_cutoff = similarity_cutoff

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

    def _get_text_from_node(
        self,
        node: Node,
        level: Optional[int] = None,
    ) -> TextChunk:
        """Query a given node.

        If node references a given document, then return the document.
        If node references a given index, then query the index.

        """
        level_str = "" if level is None else f"[Level {level}]"
        fmt_text_chunk = truncate_text(node.get_text(), 50)
        logger.debug(f">{level_str} Searching in chunk: {fmt_text_chunk}")

        response_txt = node.get_text()
        fmt_response = truncate_text(response_txt, 200)
        if self._verbose:
            print_text(f">{level_str} Got node text: {fmt_response}\n", color="blue")
        return TextChunk(response_txt)

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

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get list of tuples of node and similarity for response.

        First part of the tuple is the node.
        Second part of tuple is the distance from query to the node.
        If not applicable, it's None.
        """
        similarity_tracker = SimilarityTracker()
        nodes = self._retrieve(query_bundle, similarity_tracker=similarity_tracker)

        postprocess_info = {
            "similarity_tracker": similarity_tracker,
            "query_bundle": query_bundle,
        }
        for node_processor in self.node_preprocessors:
            nodes = node_processor.postprocess_nodes(nodes, postprocess_info)

        # TODO: create a `display` method to allow subclasses to print the Node
        return similarity_tracker.get_zipped_nodes(nodes)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        return []

    def _get_extra_info_for_response(
        self,
        nodes: List[Node],
    ) -> Optional[Dict[str, Any]]:
        """Get extra info for response."""
        return None

    def _prepare_response_builder(
        self,
        response_builder: ResponseBuilder,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]],
    ) -> None:
        """Prepare response builder and return values for query time."""
        response_builder.reset()
        for node_with_score in nodes:
            text = self._get_text_from_node(node_with_score.node)
            response_builder.add_node_as_source(
                node_with_score.node, similarity=node_with_score.score
            )
            if self._optimizer is not None:
                text = TextChunk(text=self._optimizer.optimize(query_bundle, text.text))
            response_builder.add_text_chunks([text])

        # from recursive
        if additional_source_nodes is not None:
            for node in additional_source_nodes:
                response_builder.add_node_with_score(node)

    def _prepare_response_output(
        self,
        response_builder: ResponseBuilder,
        response_str: Optional[RESPONSE_TEXT_TYPE],
        tuples: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        """Prepare response object from response string."""
        response_extra_info = self._get_extra_info_for_response(
            [node_with_score.node for node_with_score in tuples]
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

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[List[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        # prepare response builder
        self._prepare_response_builder(
            self.response_builder,
            query_bundle,
            nodes,
            additional_source_nodes=additional_source_nodes,
        )

        if self._response_mode != ResponseMode.NO_TEXT:
            response_str = self._give_response_for_nodes(
                self.response_builder, query_bundle.query_str
            )
        else:
            response_str = None

        return self._prepare_response_output(self.response_builder, response_str, nodes)

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[List[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        # define a response builder for async queries
        response_builder = ResponseBuilder(
            self._service_context,
            self.text_qa_template,
            self.refine_template,
            use_async=self._use_async,
            streaming=self._streaming,
        )
        # prepare response builder
        self._prepare_response_builder(
            response_builder,
            query_bundle,
            nodes,
            additional_source_nodes=additional_source_nodes,
        )

        if self._response_mode != ResponseMode.NO_TEXT:
            response_str = await self._agive_response_for_nodes(
                response_builder, query_bundle.query_str
            )
        else:
            response_str = None

        return self._prepare_response_output(response_builder, response_str, nodes)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: remove _query and just use query
        nodes = self.retrieve(query_bundle)
        return self.synthesize(query_bundle, nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query asynchronously."""
        # TODO: remove _query and just use query
        nodes = self.retrieve(query_bundle)
        response = await self.asynthesize(query_bundle, nodes)
        return response

    @llm_token_counter("query")
    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: support include summary
        return self._query(query_bundle)

    @llm_token_counter("query")
    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: support include summary
        return await self._aquery(query_bundle)
