import logging
from typing import List, Dict, Optional
import os

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, QueryType
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
import llama_index.core.instrumentation as instrument

from llama_index.indices.managed.dashscope import utils
from llama_index.indices.managed.dashscope.constants import (
    DASHSCOPE_DEFAULT_BASE_URL,
    RETRIEVE_PIPELINE_ENDPOINT,
    PIPELINE_SIMPLE_ENDPOINT,
)

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)


class DashScopeCloudRetriever(BaseRetriever):
    """Initialize the DashScopeCloud Retriever."""

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        dense_similarity_top_k: Optional[int] = 100,
        sparse_similarity_top_k: Optional[int] = 100,
        enable_rewrite: Optional[bool] = False,
        rewrite_model_name: Optional[str] = "conv-rewrite-qwen-1.8b",
        enable_reranking: Optional[bool] = True,
        rerank_model_name: Optional[str] = "gte-rerank-hybrid",
        rerank_min_score: Optional[float] = 0.0,
        rerank_top_n: Optional[int] = 5,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        self.index_name = index_name
        self.workspace_id = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID")
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.dense_similarity_top_k = dense_similarity_top_k
        self.sparse_similarity_top_k = sparse_similarity_top_k
        self.enable_rewrite = enable_rewrite
        self.rewrite_model_name = rewrite_model_name
        self.enable_reranking = enable_reranking
        self.rerank_model_name = rerank_model_name
        self.rerank_min_score = rerank_min_score
        self.rerank_top_n = rerank_top_n

        self.headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "utf-8",
            "X-DashScope-WorkSpace": self.workspace_id,
            "Authorization": self._api_key,
            "X-DashScope-OpenAPISource": "CloudSDK",
        }

        base_url = (
            os.environ.get("DASHSCOPE_BASE_URL", None) or DASHSCOPE_DEFAULT_BASE_URL
        )
        self.pipeline_id = utils.get_pipeline_id(
            base_url + PIPELINE_SIMPLE_ENDPOINT,
            self.headers,
            {"pipeline_name": self.index_name},
        )

        self.base_url = base_url + RETRIEVE_PIPELINE_ENDPOINT.format(
            pipeline_id=self.pipeline_id
        )
        super().__init__(callback_manager)

    @dispatcher.span
    def retrieve(
        self, str_or_query_bundle: QueryType, query_history: List[Dict] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        dispatch_event = dispatcher.get_dispatch_event()

        self._check_callback_manager()
        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle, query_history=query_history)
                nodes = self._handle_recursive_retrieval(query_bundle, nodes)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    async def _aretrieve(
        self, query_bundle: QueryBundle, query_history: List[Dict] = None
    ) -> List[NodeWithScore]:
        return self._retrieve(query_bundle, query_history=query_history)

    @dispatcher.span
    async def aretrieve(
        self, str_or_query_bundle: QueryType, query_history: List[Dict] = None
    ) -> List[NodeWithScore]:
        self._check_callback_manager()
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(
                    query_bundle=query_bundle, query_history=query_history
                )
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    def _retrieve(self, query_bundle: QueryBundle, **kwargs) -> List[NodeWithScore]:
        # init params
        params = {
            "query": query_bundle.query_str,
            "dense_similarity_top_k": self.dense_similarity_top_k,
            "sparse_similarity_top_k": self.sparse_similarity_top_k,
            "enable_rewrite": self.enable_rewrite,
            "rewrite": [
                {
                    "model_name": self.rewrite_model_name,
                    "class_name": "DashScopeTextRewrite",
                }
            ],
            "enable_reranking": self.enable_reranking,
            "rerank": [
                {
                    "model_name": self.rerank_model_name,
                }
            ],
            "rerank_min_score": self.rerank_min_score,
            "rerank_top_n": self.rerank_top_n,
        }
        # extract query_history for multi-turn query rewrite
        if "query_history" in kwargs:
            params["query_hisory"] = kwargs.get("query_history")

        response_data = utils.post(self.base_url, headers=self.headers, params=params)
        nodes = []
        for ele in response_data["nodes"]:
            text_node = TextNode.parse_obj(ele["node"])
            nodes.append(NodeWithScore(node=text_node, score=ele["score"]))
        return nodes
