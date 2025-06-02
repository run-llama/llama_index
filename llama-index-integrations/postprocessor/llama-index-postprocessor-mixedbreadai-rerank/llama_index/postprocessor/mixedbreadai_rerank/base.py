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
from mixedbread_ai.core import RequestOptions
from mixedbread_ai.client import MixedbreadAI
import httpx

dispatcher = get_dispatcher(__name__)


class MixedbreadAIRerank(BaseNodePostprocessor):
    """
    Class for reranking nodes using the mixedbread ai reranking API with models such as 'mixedbread-ai/mxbai-rerank-large-v1'.

    Args:
        top_n (int): Top N nodes to return. Defaults to 10.
        model (str): mixedbread ai model name. Defaults to "mixedbread-ai/mxbai-rerank-large-v1".
        api_key (Optional[str]): mixedbread ai API key. Defaults to None.
        max_retries (Optional[int]): Maximum number of retries for API calls. Defaults to None.
        timeout (Optional[float]): Timeout for API calls.
        httpx_client (Optional[httpx.Client]): Custom HTTPX client for synchronous requests.
        httpx_async_client (Optional[httpx.AsyncClient]): Custom HTTPX client for asynchronous requests.

    """

    model: str = Field(
        default="mixedbread-ai/mxbai-rerank-large-v1",
        description="mixedbread ai model name.",
        min_length=1,
    )
    top_n: int = Field(default=10, description="Top N nodes to return.", gt=0)

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    _request_options: Optional[RequestOptions] = PrivateAttr()

    def __init__(
        self,
        top_n: int = 10,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        api_key: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        httpx_async_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        try:
            api_key = api_key or os.environ["MXBAI_API_KEY"]
        except KeyError:
            raise ValueError(
                "Must pass in mixedbread ai API key or "
                "specify via MXBAI_API_KEY environment variable"
            )

        self._client = MixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_client
        )
        self._async_client = MixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_async_client
        )
        self._request_options = (
            RequestOptions(max_retries=max_retries) if max_retries is not None else None
        )

    @classmethod
    def class_name(cls) -> str:
        return "MixedbreadAIRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Postprocess nodes by reranking them using the mixedbread ai reranking API.

        Args:
            nodes (List[NodeWithScore]): List of nodes to rerank.
            query_bundle (Optional[QueryBundle]): Query bundle containing the query string.

        Returns:
            List[NodeWithScore]: Reranked list of nodes.

        """
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
            results = self._client.reranking(
                model=self.model,
                query=query_bundle.query_str,
                input=texts,
                top_k=self.top_n,
                return_input=False,
                request_options=self._request_options,
            )

            new_nodes = []
            for result in results.data:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
