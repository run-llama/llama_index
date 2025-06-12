from typing import Any, List, Optional

import httpx
from llama_cloud import (
    CompositeRetrievalMode,
    CompositeRetrievedTextNodeWithScore,
    RetrieverCreate,
    Retriever,
    RetrieverPipeline,
    PresetRetrievalParams,
    ReRankConfig,
)
from llama_cloud.resources.pipelines.client import OMIT

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.ingestion.api_utils import get_aclient, get_client
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.indices.managed.llama_cloud.base import LlamaCloudIndex
from llama_index.indices.managed.llama_cloud.api_utils import (
    resolve_project,
    resolve_retriever,
    page_screenshot_nodes_to_node_with_score,
)


class LlamaCloudCompositeRetriever(BaseRetriever):
    def __init__(
        self,
        # retriever identifier
        name: Optional[str] = None,
        retriever_id: Optional[str] = None,
        # project identifier
        project_name: Optional[str] = DEFAULT_PROJECT_NAME,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        # creation options
        create_if_not_exists: bool = False,
        # connection params
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        httpx_client: Optional[httpx.Client] = None,
        async_httpx_client: Optional[httpx.AsyncClient] = None,
        # composite retrieval params
        mode: Optional[CompositeRetrievalMode] = None,
        rerank_top_n: Optional[int] = None,
        rerank_config: Optional[ReRankConfig] = None,
        persisted: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Composite Retriever."""
        # initialize clients
        self._client = get_client(api_key, base_url, app_url, timeout, httpx_client)
        self._aclient = get_aclient(
            api_key, base_url, app_url, timeout, async_httpx_client
        )

        self.project = resolve_project(
            self._client, project_name, project_id, organization_id
        )

        self.name = name
        self.project_name = self.project.name
        self._persisted = persisted

        self.retriever = resolve_retriever(
            self._client, self.project, name, retriever_id, persisted
        )

        if self.retriever is None and persisted:
            if create_if_not_exists:
                self.retriever = self._client.retrievers.upsert_retriever(
                    project_id=self.project.id,
                    request=RetrieverCreate(name=self.name, pipelines=[]),
                )
            else:
                raise ValueError(
                    f"Retriever with name '{self.name}' does not exist in project."
                )

        # composite retrieval params
        self._mode = mode if mode is not None else OMIT
        self._rerank_top_n = rerank_top_n if rerank_top_n is not None else OMIT
        self._rerank_config = rerank_config if rerank_config is not None else OMIT

        super().__init__(
            callback_manager=kwargs.get("callback_manager"),
            verbose=kwargs.get("verbose", False),
        )

    @property
    def retriever_pipelines(self) -> List[RetrieverPipeline]:
        return self.retriever.pipelines or []

    def update_retriever_pipelines(
        self, pipelines: List[RetrieverPipeline]
    ) -> Retriever:
        if self._persisted:
            self.retriever = self._client.retrievers.update_retriever(
                self.retriever.id, pipelines=pipelines
            )
        else:
            # Update in-memory retriever for non-persisted case using copy
            self.retriever = self.retriever.copy(update={"pipelines": pipelines})
        return self.retriever

    def add_index(
        self,
        index: LlamaCloudIndex,
        name: Optional[str] = None,
        description: Optional[str] = None,
        preset_retrieval_parameters: Optional[PresetRetrievalParams] = None,
    ) -> Retriever:
        name = name or index.name
        preset_retrieval_parameters = (
            preset_retrieval_parameters or index.pipeline.preset_retrieval_parameters
        )
        retriever_pipeline = RetrieverPipeline(
            pipeline_id=index.id,
            name=name,
            description=description,
            preset_retrieval_parameters=preset_retrieval_parameters,
        )
        current_retriever_pipelines_by_name = {
            pipeline.name: pipeline for pipeline in (self.retriever_pipelines or [])
        }
        current_retriever_pipelines_by_name[retriever_pipeline.name] = (
            retriever_pipeline
        )
        return self.update_retriever_pipelines(
            list(current_retriever_pipelines_by_name.values())
        )

    def remove_index(self, name: str) -> bool:
        current_retriever_pipeline_names = self.retriever.pipelines or []
        new_retriever_pipelines = [
            pipeline
            for pipeline in current_retriever_pipeline_names
            if pipeline.name != name
        ]
        if len(new_retriever_pipelines) == len(current_retriever_pipeline_names):
            return False
        self.update_retriever_pipelines(new_retriever_pipelines)
        return True

    async def aupdate_retriever_pipelines(
        self, pipelines: List[RetrieverPipeline]
    ) -> Retriever:
        if self._persisted:
            self.retriever = await self._aclient.retrievers.update_retriever(
                self.retriever.id, pipelines=pipelines
            )
        else:
            # Update in-memory retriever for non-persisted case using copy
            self.retriever = self.retriever.copy(update={"pipelines": pipelines})
        return self.retriever

    async def async_add_index(
        self,
        index: LlamaCloudIndex,
        name: Optional[str] = None,
        description: Optional[str] = None,
        preset_retrieval_parameters: Optional[PresetRetrievalParams] = None,
    ) -> Retriever:
        name = name or index.name
        preset_retrieval_parameters = (
            preset_retrieval_parameters or index.pipeline.preset_retrieval_parameters
        )
        retriever_pipeline = RetrieverPipeline(
            pipeline_id=index.id,
            name=name,
            description=description,
            preset_retrieval_parameters=preset_retrieval_parameters,
        )
        current_retriever_pipelines_by_name = {
            pipeline.name: pipeline for pipeline in (self.retriever_pipelines or [])
        }
        current_retriever_pipelines_by_name[retriever_pipeline.name] = (
            retriever_pipeline
        )
        return await self.aupdate_retriever_pipelines(
            list(current_retriever_pipelines_by_name.values())
        )

    async def aremove_index(self, name: str) -> bool:
        current_retriever_pipeline_names = self.retriever.pipelines or []
        new_retriever_pipelines = [
            pipeline
            for pipeline in current_retriever_pipeline_names
            if pipeline.name != name
        ]
        if len(new_retriever_pipelines) == len(current_retriever_pipeline_names):
            return False
        await self.aupdate_retriever_pipelines(new_retriever_pipelines)
        return True

    def _result_nodes_to_node_with_score(
        self, composite_retrieval_node: CompositeRetrievedTextNodeWithScore
    ) -> NodeWithScore:
        return NodeWithScore(
            node=TextNode(
                id=composite_retrieval_node.node.id,
                text=composite_retrieval_node.node.text,
                metadata=composite_retrieval_node.node.metadata,
            ),
            score=composite_retrieval_node.score,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        mode: Optional[CompositeRetrievalMode] = None,
        rerank_top_n: Optional[int] = None,
        rerank_config: Optional[ReRankConfig] = None,
    ) -> List[NodeWithScore]:
        mode = mode if mode is not None else self._mode

        rerank_top_n = rerank_top_n if rerank_top_n is not None else self._rerank_top_n
        rerank_config = (
            rerank_config if rerank_config is not None else self._rerank_config
        )

        if self._persisted:
            result = self._client.retrievers.retrieve(
                self.retriever.id,
                mode=mode,
                rerank_top_n=rerank_top_n,
                rerank_config=rerank_config,
                query=query_bundle.query_str,
            )
        else:
            result = self._client.retrievers.direct_retrieve(
                project_id=self.project.id,
                mode=mode,
                rerank_top_n=rerank_top_n,
                rerank_config=rerank_config,
                query=query_bundle.query_str,
                pipelines=self.retriever.pipelines,
            )
        node_w_scores = [
            self._result_nodes_to_node_with_score(node) for node in result.nodes
        ]
        image_nodes_w_scores = page_screenshot_nodes_to_node_with_score(
            self._client, result.image_nodes, self.retriever.project_id
        )
        return sorted(
            node_w_scores + image_nodes_w_scores, key=lambda x: x.score, reverse=True
        )

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
        mode: Optional[CompositeRetrievalMode] = None,
        rerank_top_n: Optional[int] = None,
    ) -> List[NodeWithScore]:
        mode = mode if mode is not None else self._mode

        rerank_top_n = rerank_top_n if rerank_top_n is not None else self._rerank_top_n
        rerank_config = (
            rerank_config if rerank_config is not None else self._rerank_config
        )

        if self._persisted:
            result = await self._aclient.retrievers.retrieve(
                self.retriever.id,
                mode=mode,
                rerank_config=rerank_config,
                rerank_top_n=rerank_top_n,
                query=query_bundle.query_str,
            )
        else:
            result = await self._aclient.retrievers.direct_retrieve(
                project_id=self.project.id,
                mode=mode,
                rerank_top_n=rerank_top_n,
                rerank_config=rerank_config,
                query=query_bundle.query_str,
                pipelines=self.retriever.pipelines,
            )
        node_w_scores = [
            self._result_nodes_to_node_with_score(node) for node in result.nodes
        ]
        image_nodes_w_scores = page_screenshot_nodes_to_node_with_score(
            self._aclient, result.image_nodes, self.retriever.project_id
        )
        return sorted(
            node_w_scores + image_nodes_w_scores, key=lambda x: x.score, reverse=True
        )
