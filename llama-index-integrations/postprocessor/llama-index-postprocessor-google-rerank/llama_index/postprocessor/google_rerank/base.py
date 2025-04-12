import os
import json
from enum import Enum
from typing import Any, List, Optional, TypedDict

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

import google.auth
from google.cloud import discoveryengine_v1 as discoveryengine


dispatcher = get_dispatcher(__name__)

class VertexAIConfig(TypedDict):
    credentials: Optional[google.auth.credentials.Credentials] = None
    project: Optional[str] = None
    location: Optional[str] = None

class Models(str, Enum):
    SEMANTIC_RERANK_512_003 = "semantic-ranker-512-003"


class GoogleRerank(BaseNodePostprocessor):
    top_n: int = Field(default=2, description="Top N nodes to return.")
    rerank_model_name: str = Field(
        default=Models.SEMANTIC_RERANK_512_003.value,
        description="The modelId of the VertexAI model to use.",
    )
    _client: discoveryengine.RankServiceClient = PrivateAttr()
    _ranking_config: str = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        rerank_model_name: str = Models.SEMANTIC_RERANK_512_003.value,
        client: Optional[discoveryengine.RankServiceClient] = None,
        vertexai_config: Optional[VertexAIConfig] = None,
        ranking_config: Optional[Any] = "default_ranking_config",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_n = top_n
        self.rerank_model_name = rerank_model_name

        project = (vertexai_config or {}).get("project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT", None
        )
        credentials = json.loads((vertexai_config or {}).get("credentials") or os.getenv(
            "GOOGLE_CLOUD_CREDENTIALS", None
        ))
        location = (vertexai_config or {}).get("location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", None
        )

        if client is not None:
            self._client = client
            self._ranking_config = client.ranking_config_path(
                project=project,
                location=location,
                ranking_config=ranking_config,
            )

        elif vertexai_config is not None:
            self._client = discoveryengine.RankServiceClient(credentials=credentials)
            self._ranking_config = self._client.ranking_config_path(
                project=project,
                location=location,
                ranking_config=ranking_config,
            )

    @classmethod
    def class_name(cls) -> str:
        return "GoogleRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if dispatcher:
            dispatcher.event(
                ReRankStartEvent(
                    query=query_bundle,
                    nodes=nodes,
                    top_n=self.top_n,
                    model_name=self.rerank_model_name,
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
                EventPayload.MODEL_NAME: self.rerank_model_name,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:

            # Prepare the text sources for Google Reranker
            text_sources = []
            for index, node in enumerate(nodes):
                text_sources.append(
                    discoveryengine.RankingRecord(
                        id=str(index),
                        content=node.node.get_content(metadata_mode=MetadataMode.EMBED),
                    ),
                )
            # change top_n if the number of nodes is less than top_n
            if len(nodes) < self.top_n:
                self.top_n = len(nodes)

            try:
                request = discoveryengine.RankRequest(
                    ranking_config=self._ranking_config,
                    model=self.rerank_model_name,
                    top_n=self.top_n,
                    query=query_bundle.query_str,
                    records=text_sources
                )
                response = self._client.rank(request=request)

                results = response["records"]
            except Exception as e:
                raise RuntimeError(f"Failed to invoke VertexAI model: {e}")

            new_nodes = []
            for result in results:
                index = int(result["id"])
                relevance_score = result.get("score", 0.0)
                new_node_with_score = NodeWithScore(
                    node=nodes[index].node,
                    score=relevance_score,
                )
                new_nodes.append(new_node_with_score)

            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
