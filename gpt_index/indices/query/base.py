"""Base query classes."""

import logging
from abc import ABC
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.docstore import DocumentStore
from gpt_index.indices.postprocessor.node import (
    BaseNodePostprocessor,
)
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.synthesis import ResponseSynthesizer
from gpt_index.indices.response.builder import ResponseMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.optimization.optimizer import BaseTokenUsageOptimizer
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import (
    RESPONSE_TYPE,
)
from gpt_index.token_counter.token_counter import llm_token_counter

# to prevent us from having to remove all instances of v2 later
IndexStruct = V2IndexStruct
IS = TypeVar("IS", bound=V2IndexStruct)

logger = logging.getLogger(__name__)


class BaseGPTIndexQuery(Generic[IS], ABC):
    """Base LlamaIndex Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    Args:
        service_context (ServiceContext): service context container (contains components
            like LLMPredictor, PromptHelper).
        response_mode (ResponseMode): Optional ResponseMode. If not provided, will
            use the default ResponseMode.
        text_qa_template (QuestionAnswerPrompt): Optional QuestionAnswerPrompt object.
            If not provided, will use the default QuestionAnswerPrompt.
        refine_template (RefinePrompt): Optional RefinePrompt object. If not provided,
            will use the default RefinePrompt.
        streaming (bool): Optional bool. If True, will return a StreamingResponse
            object. If False, will return a Response object.

    """

    def __init__(
        self,
        index_struct: IS,
        service_context: ServiceContext,
        response_synthesizer: ResponseSynthesizer,
        docstore: Optional[DocumentStore] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        if docstore is None:
            raise ValueError("docstore must be provided.")

        self._validate_index_struct(index_struct)

        self._index_struct = index_struct
        self._docstore = docstore
        self._service_context = service_context
        self._response_synthesizer = response_synthesizer

        # initialize logger with metadata
        if self._service_context.llama_logger is not None:
            self._service_context.llama_logger.set_metadata(
                {
                    "index_type": self._index_struct.get_type(),
                    "index_id": self._index_struct.index_id,
                }
            )

        self.node_preprocessors = node_postprocessors or []
        self._verbose = verbose
    

    @classmethod
    def from_args(cls, 
        index_struct: IS,
        service_context: ServiceContext,
        response_synthesizer: ResponseSynthesizer,
        docstore: Optional[DocumentStore] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        response_kwargs: Optional[Dict] = None,
        use_async: bool = False,
        streaming: bool = False,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
    ):
        response_synthesizer = ResponseSynthesizer.from_args(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            response_mode=response_mode,
            response_kwargs=response_kwargs,
            use_async=use_async,
            streaming=streaming,
            optimizer=optimizer
        )
        return cls(
            index_struct,
            service_context,
            response_synthesizer,
            docstore,
            node_postprocessors,
            verbose,
        )

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

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

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        # TODO: remove _query and just use query
        nodes = self.retrieve(query_bundle)
        return self._synthesizer.synthesize(query_bundle, nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query asynchronously."""
        # TODO: remove _query and just use query
        nodes = self.retrieve(query_bundle)
        response = await self._synthesizer.asynthesize(query_bundle, nodes)
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
