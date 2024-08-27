import requests
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)


class XinferenceRerank(BaseNodePostprocessor):
    """Class for Xinference Rerank."""

    top_n: int = Field(
        default=5,
        description="The number of nodes to return.",
    )
    model: str = Field(
        default="bge-reranker-base",
        description="The Xinference model uid to use.",
    )
    base_url: str = Field(
        default="http://localhost:9997",
        description="The Xinference base url to use.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "XinferenceRerank"

    def get_query_str(self, query):
        return query.query_str if isinstance(query, QueryBundle) else query

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
                model_name=self.model,
            )
        )
        if query_bundle is None:
            raise ValueError("Missing query bundle.")
        if len(nodes) == 0:
            return []
        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: self.get_query_str(query_bundle),
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            headers = {"Content-Type": "application/json"}
            json_data = {
                "model": self.model,
                "query": self.get_query_str(query_bundle),
                "documents": [
                    node.node.get_content(metadata_mode=MetadataMode.EMBED)
                    for node in nodes
                ],
            }
            response = requests.post(
                url=f"{self.base_url}/v1/rerank", headers=headers, json=json_data
            )
            response.encoding = "utf-8"
            if response.status_code != 200:
                raise Exception(
                    f"Xinference call failed with status code {response.status_code}."
                    f"Details: {response.text}"
                )
            rerank_nodes = [
                NodeWithScore(
                    node=nodes[result["index"]].node, score=result["relevance_score"]
                )
                for result in response.json()["results"][: self.top_n]
            ]
            event.on_end(payload={EventPayload.NODES: rerank_nodes})
        dispatcher.event(ReRankEndEvent(nodes=rerank_nodes))
        return rerank_nodes
