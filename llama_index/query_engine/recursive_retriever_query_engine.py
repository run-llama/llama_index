from typing import Dict, List, Optional, Tuple, Any, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import TextNode, IndexNode, NodeWithScore
from llama_index.bridge.langchain import print_text
from llama_index.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.response.type import ResponseMode
from llama_index.optimization.optimizer import BaseTokenUsageOptimizer
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.indices.base_retriever import BaseRetriever


class RecursiveRetrieverQueryEngine(BaseQueryEngine):
    """Recursive Retriever query engine.

    Operates specifically over a RecursiveRetriever.

    NOTE: This is similar to plugging in a RecursiveRetriever into a RetrieverQueryEngine,
    but with one main difference + syntatic sugar for convenience.
        - main difference is that this query engine can natively include all
            source nodes from sub-query engines in the response.

    Args:
        root_id (str): The root id of the query graph.
        query_engines (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
            id to query engines.

    """

    def __init__(
        self,
        retriever: RecursiveRetriever,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = (
            response_synthesizer
            or ResponseSynthesizer.from_args(callback_manager=callback_manager)
        )
        super().__init__(callback_manager)

    @classmethod
    def from_args(
        cls,
        root_id: str,
        retriever_dict: Dict[str, BaseRetriever],
        query_engine_dict: Optional[Dict[str, BaseQueryEngine]] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any
    ) -> "RetrieverQueryEngine":
        """Initialize a RetrieverQueryEngine object."

        Args:
            root_id (str): The root id of the query graph.
            retriever_dict (Optional[Dict[str, BaseRetriever]]): A dictionary
                of id to retrievers.
            query_engine_dict (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
                id to query engines.
            service_context (Optional[ServiceContext]): A ServiceContext object.
            **kwargs: Additional keyword arguments for ResponseSynthesizer.

        """
        callback_manager = (
            service_context.callback_manager if service_context else CallbackManager([])
        )
        retriever = RecursiveRetriever(
            root_id=root_id,
            retriever_dict=retriever_dict,
            query_engine_dict=query_engine_dict,
            callback_manager=callback_manager,
        )
        response_synthesizer = ResponseSynthesizer.from_args(
            service_context=service_context,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_id = self.callback_manager.on_event_start(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        )

        retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
        nodes, additional_nodes = self._retriever.retrieve_all(query_bundle)
        self.callback_manager.on_event_end(
            CBEventType.RETRIEVE,
            payload={EventPayload.NODES: nodes},
            event_id=retrieve_id,
        )

        response = self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_nodes,
        )

        self.callback_manager.on_event_end(
            CBEventType.QUERY,
            payload={EventPayload.RESPONSE: response},
            event_id=query_id,
        )
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_id = self.callback_manager.on_event_start(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        )

        retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
        nodes, additional_nodes = self._retriever.retrieve_all(query_bundle)
        self.callback_manager.on_event_end(
            CBEventType.RETRIEVE,
            payload={EventPayload.NODES: nodes},
            event_id=retrieve_id,
        )

        response = await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_nodes,
        )

        self.callback_manager.on_event_end(
            CBEventType.QUERY,
            payload={EventPayload.RESPONSE: response},
            event_id=query_id,
        )
        return response

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever
