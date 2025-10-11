import os
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode


dispatcher = get_dispatcher(__name__)


class ContextualRerank(BaseNodePostprocessor):
    """
    Contextual Reranking model.

    Args:
        model: str = Field(description="Contextual Reranking model name. Default is 'ctxl-rerank-en-v1-instruct'.")
        top_n: int = Field(description="Top N nodes to return.")
        base_url: Optional[str] = Field(description="Contextual base url.", default=None)

    """

    model: str = Field(description="Contextual Reranking model name.")
    top_n: int = Field(description="Top N nodes to return.")
    base_url: Optional[str] = Field(description="Contextual base url.", default=None)

    _client: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "ctxl-rerank-en-v1-instruct",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        try:
            api_key = api_key or os.environ["CONTEXTUAL_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in contextual api key or "
                "specify via CONTEXTUAL_API_KEY environment variable "
            )
        try:
            from contextual import ContextualAI
        except ImportError:
            raise ImportError(
                "Cannot import Contextual client package, please `pip install contextual-client`."
            )

        if client is not None:
            self._client = client
        else:
            try:
                self._client = ContextualAI(api_key=api_key, base_url=base_url)
            except Exception as e:
                raise ValueError(f"Failed to create Contextual client: {e}")

    @classmethod
    def class_name(cls) -> str:
        return "ContextualRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
            results = self._client.rerank.create(
                model=self.model,
                top_n=self.top_n,
                query=query_bundle.query_str,
                documents=texts,
            )

            new_nodes = []
            for result in results.results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
