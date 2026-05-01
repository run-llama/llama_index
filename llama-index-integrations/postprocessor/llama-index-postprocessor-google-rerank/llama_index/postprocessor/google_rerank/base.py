"""Google Rerank postprocessor using Discovery Engine Ranking API."""

import os
from typing import Any, List, Optional

import google.auth

from google.cloud import discoveryengine_v1 as discoveryengine

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)

DEFAULT_MODEL = "semantic-ranker-default-004"
DEFAULT_LOCATION = "global"
DEFAULT_RANKING_CONFIG = "default_ranking_config"


class GoogleRerank(BaseNodePostprocessor):
    """
    Google Rerank postprocessor.

    Uses Google's Discovery Engine Ranking API to rerank nodes
    based on query relevance.
    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="The ranking model to use.",
    )
    top_n: int = Field(default=2, description="Top N nodes to return.")
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Google Cloud project ID. Falls back to GOOGLE_CLOUD_PROJECT "
            "env var, then Application Default Credentials."
        ),
    )
    location: str = Field(
        default=DEFAULT_LOCATION,
        description="Google Cloud location for the ranking config.",
    )
    ranking_config: str = Field(
        default=DEFAULT_RANKING_CONFIG,
        description="Name of the ranking config resource.",
    )

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_n: int = 2,
        project_id: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        ranking_config: str = DEFAULT_RANKING_CONFIG,
        credentials: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            top_n=top_n,
            project_id=project_id,
            location=location,
            ranking_config=ranking_config,
            **kwargs,
        )

        # Resolve project_id
        if self.project_id is None:
            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if self.project_id is None:
            _, resolved_project = google.auth.default()
            self.project_id = resolved_project

        self._client = discoveryengine.RankServiceClient(credentials=credentials)
        self._async_client = discoveryengine.RankServiceAsyncClient(
            credentials=credentials
        )

    @classmethod
    def class_name(cls) -> str:
        return "GoogleRerank"

    def _build_ranking_config_path(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/rankingConfigs/{self.ranking_config}"
        )

    def _build_records(self, nodes: List[NodeWithScore]) -> list:
        records = []
        for i, node in enumerate(nodes):
            content = node.node.get_content(metadata_mode=MetadataMode.EMBED)
            records.append(
                discoveryengine.RankingRecord(
                    id=str(i),
                    content=content,
                )
            )
        return records

    @dispatcher.span
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
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        records = self._build_records(nodes)
        top_n = min(self.top_n, len(nodes))

        request = discoveryengine.RankRequest(
            ranking_config=self._build_ranking_config_path(),
            model=self.model,
            top_n=top_n,
            query=query_bundle.query_str,
            records=records,
        )

        response = self._client.rank(request=request)

        new_nodes = []
        for record in response.records:
            index = int(record.id)
            new_nodes.append(
                NodeWithScore(
                    node=nodes[index].node,
                    score=record.score,
                )
            )

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes

    @dispatcher.span
    async def _apostprocess_nodes(
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
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        records = self._build_records(nodes)
        top_n = min(self.top_n, len(nodes))

        request = discoveryengine.RankRequest(
            ranking_config=self._build_ranking_config_path(),
            model=self.model,
            top_n=top_n,
            query=query_bundle.query_str,
            records=records,
        )

        response = await self._async_client.rank(request=request)

        new_nodes = []
        for record in response.records:
            index = int(record.id)
            new_nodes.append(
                NodeWithScore(
                    node=nodes[index].node,
                    score=record.score,
                )
            )

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
