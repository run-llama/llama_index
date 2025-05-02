import time
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    from alibabacloud_searchplat20240529.models import (
        GetDocumentRankRequest,
        GetDocumentRankResponse,
        GetDocumentRankResponseBodyResultScores,
    )
    from alibabacloud_tea_openapi.models import Config as AISearchConfig
    from alibabacloud_searchplat20240529.client import Client
    from Tea.exceptions import TeaException
except ImportError:
    raise ImportError(
        "Could not import alibabacloud_searchplat20240529 python package. "
        "Please install it with `pip install alibabacloud-searchplat20240529`."
    )

dispatcher = get_dispatcher(__name__)


def retry_decorator(func, wait_seconds: int = 1):
    def wrap(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    time.sleep(wait_seconds)
                else:
                    raise

    return wrap


class AlibabaCloudAISearchRerank(BaseNodePostprocessor):
    """
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/ranker-api-details`.
    """

    _client: Client = PrivateAttr()

    aisearch_api_key: str = Field(default=None, exclude=True)
    endpoint: str = None

    service_id: str = "ops-bge-reranker-larger"
    workspace_name: str = "default"
    top_n: int = 3
    batch_size: int = 16

    def __init__(
        self, endpoint: str = None, aisearch_api_key: str = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.aisearch_api_key = get_from_param_or_env(
            "aisearch_api_key", aisearch_api_key, "AISEARCH_API_KEY"
        )
        self.endpoint = get_from_param_or_env("endpoint", endpoint, "AISEARCH_ENDPOINT")

        config = AISearchConfig(
            bearer_token=self.aisearch_api_key,
            endpoint=self.endpoint,
            protocol="http",
        )

        self._client = Client(config=config)

    @classmethod
    def class_name(cls) -> str:
        return "AlibabaCloudAISearchRerank"

    @retry_decorator
    def _rerank_one_batch(
        self, query: str, texts: List[str]
    ) -> List[GetDocumentRankResponseBodyResultScores]:
        request = GetDocumentRankRequest(docs=texts, query=query)
        response: GetDocumentRankResponse = self._client.get_document_rank(
            workspace_name=self.workspace_name,
            service_id=self.service_id,
            request=request,
        )
        return response.body.result.scores

    def _rerank(
        self, query: str, texts: List[str], top_n: int
    ) -> List[GetDocumentRankResponseBodyResultScores]:
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_scores = self._rerank_one_batch(query, texts[i : i + self.batch_size])
            for score in batch_scores:
                score.index = i + score.index
            scores.extend(batch_scores)
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_n]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.service_id,
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
                EventPayload.MODEL_NAME: self.service_id,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]
            results = self._rerank(
                query=query_bundle.query_str,
                texts=texts,
                top_n=self.top_n,
            )

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
