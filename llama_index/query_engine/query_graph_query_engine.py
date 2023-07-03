# from typing import Dict, List, Optional, Tuple, Any

# from llama_index.callbacks.base import CallbackManager
# from llama_index.callbacks.schema import CBEventType, EventPayload
# from llama_index.indices.composability.graph import ComposableGraph
# from llama_index.indices.query.base import BaseQueryEngine
# from llama_index.indices.query.schema import QueryBundle
# from llama_index.response.schema import RESPONSE_TYPE
# from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
# from llama_index.schema import TextNode, IndexNode, NodeWithScore
# from llama_index.bridge.langchain import print_text
# from llama_index.retrievers.recursive_retriever import RecursiveRetriever
# from llama_index.indices.query.response_synthesis import ResponseSynthesizer
# from llama_index.indices.service_context import ServiceContext
# from llama_index.indices.postprocessor.types import BaseNodePostprocessor
# from llama_index.indices.response.type import ResponseMode
# from llama_index.optimization.optimizer import BaseTokenUsageOptimizer
# from llama_index.prompts.prompts import (
#     QuestionAnswerPrompt,
#     RefinePrompt,
#     SimpleInputPrompt,
# )
# from llama_index.indices.base_retriever import BaseRetriever


# # class QueryGraphQueryEngine(BaseQueryEngine):
# #     """Query graph query engine.

# #     This query engine can operate over a query graph.
# #     It can take in query engines for its sub-indices.

# #     If a query engine is a RetrieverQueryEngine, then it will
# #     first retrieve nodes. If any of the nodes are IndexNodes,
# #     then it will recursively query the query engine referenced by
# #     the index node. It will then synthesize the nodes.

# #     If the query engine is not a RetrieverQueryEngine, then it will
# #     simply query the query engine.

# #     Args:
# #         root_id (str): The root id of the query graph.
# #         query_engines (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
# #             id to query engines.

# #     """

# #     def __init__(
# #         self,
# #         root_id: str,
# #         query_engine_dict: Dict[str, BaseQueryEngine],
# #         callback_manager: Optional[CallbackManager] = None,
# #         query_response_tmpl: Optional[str] = None,
# #         verbose: bool = False,
# #     ) -> None:
# #         """Init params."""
# #         self._root_id = root_id
# #         self._query_engine_dict = query_engine_dict or {}
# #         self._query_response_tmpl = query_response_tmpl or DEFAULT_QUERY_RESPONSE_TMPL
# #         self._verbose = verbose
# #         super().__init__(callback_manager)

# #     def _query_retrieved_node(
# #         self, query_bundle: QueryBundle, node_with_score: NodeWithScore
# #     ) -> Tuple[NodeWithScore, List[NodeWithScore]]:
# #         """Query for retrieved nodes.

# #         If node is an IndexNode, then recursively query the query engine.
# #         If node is a TextNode, then simply return the node.

# #         """
# #         node = node_with_score.node
# #         if isinstance(node, IndexNode):
# #             if self._verbose:
# #                 print_text(
# #                     "Retrieved query engine with id, entering: " f"{node.index_id}\n",
# #                     color="pink",
# #                 )
# #             sub_resp = self._query_rec(query_bundle, query_id=node.index_id)
# #             if self._verbose:
# #                 print_text(
# #                     f"Got response: {str(sub_resp)}\n",
# #                     color="green",
# #                 )
# #             # format with both the query and the response
# #             node_text = self._query_response_tmpl.format(
# #                 query_str=query_bundle.query_str, response=str(sub_resp)
# #             )
# #             node = TextNode(text=node_text)
# #             node_to_add = NodeWithScore(node=node, score=1.0)
# #             additional_nodes = sub_resp.source_nodes
# #         else:
# #             assert isinstance(node, TextNode)
# #             if self._verbose:
# #                 print_text(
# #                     "Retrieving text node: " f"{node.get_content()}\n",
# #                     color="pink",
# #                 )
# #             node_to_add = node_with_score
# #             additional_nodes = []
# #         return node_to_add, additional_nodes

# #     def _query_rec(
# #         self, query_bundle: QueryBundle, query_id: Optional[str] = None
# #     ) -> RESPONSE_TYPE:
# #         """Query recursively."""
# #         if self._verbose:
# #             print_text(
# #                 f"Querying with query id {query_id}: {query_bundle.query_str}\n",
# #                 color="blue",
# #             )
# #         query_id = query_id or self._root_id
# #         query_engine = self._query_engine_dict[query_id]
# #         if isinstance(query_engine, RetrieverQueryEngine):
# #             retrieve_event_id = self.callback_manager.on_event_start(
# #                 CBEventType.RETRIEVE
# #             )
# #             nodes = query_engine.retrieve(query_bundle)
# #             self.callback_manager.on_event_end(
# #                 CBEventType.RETRIEVE,
# #                 payload={EventPayload.NODES: nodes},
# #                 event_id=retrieve_event_id,
# #             )
# #             nodes_for_synthesis = []
# #             additional_source_nodes = []
# #             for node_with_score in nodes:
# #                 node_to_add, node_additional_sources = self._query_retrieved_node(
# #                     query_bundle, node_with_score
# #                 )
# #                 nodes_for_synthesis.append(node_to_add)
# #                 additional_source_nodes.extend(node_additional_sources)
# #             response = query_engine.synthesize(
# #                 query_bundle, nodes_for_synthesis, additional_source_nodes
# #             )
# #         else:
# #             response = query_engine.query(query_bundle)

# #         if self._verbose:
# #             print_text(
# #                 f"Got response for query id {query_id}: {str(response)}\n",
# #                 color="blue",
# #             )
# #         return response

# #     async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
# #         return self._query_rec(query_bundle, query_id=None)

# #     def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
# #         return self._query_rec(query_bundle, query_id=None)


# class RecursiveRetrieverQueryEngine(BaseQueryEngine):
#     """Recursive Retriever query engine.

#     Operates specifically over a RecursiveRetriever.

#     NOTE: This is similar to plugging in a RecursiveRetriever into a RetrieverQueryEngine,
#     but with one main difference + syntatic sugar for convenience.
#         - main difference is that this query engine can natively include all
#             source nodes from sub-query engines in the response.

#     Args:
#         root_id (str): The root id of the query graph.
#         query_engines (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
#             id to query engines.

#     """

#     def __init__(
#         self,
#         retriever: RecursiveRetriever,
#         response_synthesizer: Optional[ResponseSynthesizer] = None,
#         callback_manager: Optional[CallbackManager] = None,
#     ) -> None:
#         self._retriever = retriever
#         self._response_synthesizer = (
#             response_synthesizer
#             or ResponseSynthesizer.from_args(callback_manager=callback_manager)
#         )
#         super().__init__(callback_manager)

#     @classmethod
#     def from_args(
#         cls,
#         root_id: str,
#         retriever_dict: Dict[str, BaseRetriever],
#         query_engine_dict: Optional[Dict[str, BaseQueryEngine]] = None,
#         service_context: Optional[ServiceContext] = None,
#         node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
#         verbose: bool = False,
#         # response synthesizer args
#         response_mode: ResponseMode = ResponseMode.COMPACT,
#         text_qa_template: Optional[QuestionAnswerPrompt] = None,
#         refine_template: Optional[RefinePrompt] = None,
#         simple_template: Optional[SimpleInputPrompt] = None,
#         response_kwargs: Optional[Dict] = None,
#         use_async: bool = False,
#         streaming: bool = False,
#         optimizer: Optional[BaseTokenUsageOptimizer] = None,
#         # class-specific args
#         **kwargs: Any,
#     ) -> "RetrieverQueryEngine":
#         """Initialize a RetrieverQueryEngine object."

#         Args:
#             retriever (BaseRetriever): A retriever object.
#             service_context (Optional[ServiceContext]): A ServiceContext object.
#             node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
#                 node postprocessors.
#             verbose (bool): Whether to print out debug info.
#             response_mode (ResponseMode): A ResponseMode object.
#             text_qa_template (Optional[QuestionAnswerPrompt]): A QuestionAnswerPrompt
#                 object.
#             refine_template (Optional[RefinePrompt]): A RefinePrompt object.
#             simple_template (Optional[SimpleInputPrompt]): A SimpleInputPrompt object.
#             response_kwargs (Optional[Dict]): A dict of response kwargs.
#             use_async (bool): Whether to use async.
#             streaming (bool): Whether to use streaming.
#             optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
#                 object.

#         """
#         callback_manager = (
#             service_context.callback_manager if service_context else CallbackManager([])
#         )
#         retriever = RecursiveRetriever(
#             root_id=root_id,
#             retriever_dict=retriever_dict,
#             query_engine_dict=query_engine_dict,
#             callback_manager=callback_manager,
#         )
#         response_synthesizer = ResponseSynthesizer.from_args(
#             service_context=service_context,
#             text_qa_template=text_qa_template,
#             refine_template=refine_template,
#             simple_template=simple_template,
#             response_mode=response_mode,
#             response_kwargs=response_kwargs,
#             use_async=use_async,
#             streaming=streaming,
#             optimizer=optimizer,
#             node_postprocessors=node_postprocessors,
#             verbose=verbose,
#         )

#         return cls(
#             retriever=retriever,
#             response_synthesizer=response_synthesizer,
#             callback_manager=callback_manager,
#         )

#     def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         return self._retriever.retrieve(query_bundle)

#     def synthesize(
#         self,
#         query_bundle: QueryBundle,
#         nodes: List[NodeWithScore],
#         additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
#     ) -> RESPONSE_TYPE:
#         return self._response_synthesizer.synthesize(
#             query_bundle=query_bundle,
#             nodes=nodes,
#             additional_source_nodes=additional_source_nodes,
#         )

#     async def asynthesize(
#         self,
#         query_bundle: QueryBundle,
#         nodes: List[NodeWithScore],
#         additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
#     ) -> RESPONSE_TYPE:
#         return await self._response_synthesizer.asynthesize(
#             query_bundle=query_bundle,
#             nodes=nodes,
#             additional_source_nodes=additional_source_nodes,
#         )

#     def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
#         """Answer a query."""
#         query_id = self.callback_manager.on_event_start(
#             CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
#         )

#         retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
#         nodes = self._retriever.retrieve(query_bundle)
#         self.callback_manager.on_event_end(
#             CBEventType.RETRIEVE,
#             payload={EventPayload.NODES: nodes},
#             event_id=retrieve_id,
#         )

#         response = self._response_synthesizer.synthesize(
#             query_bundle=query_bundle,
#             nodes=nodes,
#         )

#         self.callback_manager.on_event_end(
#             CBEventType.QUERY,
#             payload={EventPayload.RESPONSE: response},
#             event_id=query_id,
#         )
#         return response

#     async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
#         """Answer a query."""
#         query_id = self.callback_manager.on_event_start(
#             CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
#         )

#         retrieve_id = self.callback_manager.on_event_start(CBEventType.RETRIEVE)
#         nodes = self._retriever.retrieve(query_bundle)
#         self.callback_manager.on_event_end(
#             CBEventType.RETRIEVE,
#             payload={EventPayload.NODES: nodes},
#             event_id=retrieve_id,
#         )

#         response = await self._response_synthesizer.asynthesize(
#             query_bundle=query_bundle,
#             nodes=nodes,
#         )

#         self.callback_manager.on_event_end(
#             CBEventType.QUERY,
#             payload={EventPayload.RESPONSE: response},
#             event_id=query_id,
#         )
#         return response

#     @property
#     def retriever(self) -> BaseRetriever:
#         """Get the retriever object."""
#         return self._retriever
