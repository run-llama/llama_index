from typing import Any, List, Optional

import httpx
from llama_cloud import (
    CompositeRetrievalMode,
    CompositeRetrievedTextNodeWithScore,
    RetrieverCreate,
    Retriever,
    RetrieverPipeline,
    PresetRetrievalParams,
)
from llama_cloud.resources.pipelines.client import OMIT

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.ingestion.api_utils import get_aclient, get_client
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.indices.managed.llama_cloud.base import LlamaCloudIndex
from llama_index.indices.managed.llama_cloud.api_utils import (
    resolve_project,
    image_nodes_to_node_with_score,
)


class LlamaCloudCompositeRetriever(BaseRetriever):
    def __init__(
        self,
        # index identifier
        name: Optional[str] = None,
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
        **kwargs: Any,
    ) -> None:
        """Initialize the Composite Retriever."""
        if sum([bool(name), bool(project_id)]) != 1:
            raise ValueError(
                "Exactly one of `name` or `project_id` must be provided to identify the index."
            )

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

        # TODO: Refactor to use ?name=x query param once that is released in python client
        project_retrievers = self._client.retrievers.list_retrievers(
            project_id=self.project.id
        )
        self.retriever = next(
            (
                retriever
                for retriever in project_retrievers
                if retriever.name == self.name
            ),
            None,
        )
        if self.retriever is None:
            if create_if_not_exists:
                self.retriever = self._client.retrievers.upsert_retriever(
                    project_id=self.project.id,
                    request=RetrieverCreate(name=self.name, pipelines=[]),
                )
            else:
                raise ValueError(
                    f"Retriever with name '{self.name}' does not exist in project '{self.project_name}'."
                )

        # composite retrieval params
        self._mode = mode if mode is not None else OMIT
        self._rerank_top_n = rerank_top_n if rerank_top_n is not None else OMIT

        super().__init__(
            callback_manager=kwargs.get("callback_manager", None),
            verbose=kwargs.get("verbose", False),
        )

    @property
    def retriever_pipelines(self) -> List[RetrieverPipeline]:
        return self.retriever.pipelines or []

    def update_retriever_pipelines(
        self, pipelines: List[RetrieverPipeline]
    ) -> Retriever:
        self.retriever = self._client.retrievers.update_retriever(
            self.retriever.id, pipelines=pipelines
        )
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
        current_retriever_pipelines_by_name[
            retriever_pipeline.name
        ] = retriever_pipeline
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
        self.retriever = await self._aclient.retrievers.update_retriever(
            self.retriever.id, pipelines=pipelines
        )
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
        current_retriever_pipelines_by_name[
            retriever_pipeline.name
        ] = retriever_pipeline
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
    ) -> List[NodeWithScore]:
        mode = mode if mode is not None else self._mode
        rerank_top_n = rerank_top_n if rerank_top_n is not None else self._rerank_top_n
        result = self._client.retrievers.retrieve(
            self.retriever.id,
            mode=mode,
            rerank_top_n=rerank_top_n,
            query=query_bundle.query_str,
        )
        node_w_scores = [
            self._result_nodes_to_node_with_score(node) for node in result.nodes
        ]
        image_nodes_w_scores = image_nodes_to_node_with_score(
            self._client, result.image_nodes, self.project.id
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
        result = await self._aclient.retrievers.retrieve(
            self.retriever.id,
            mode=mode,
            rerank_top_n=rerank_top_n,
            query=query_bundle.query_str,
        )
        node_w_scores = [
            self._result_nodes_to_node_with_score(node) for node in result.nodes
        ]
        image_nodes_w_scores = image_nodes_to_node_with_score(
            self._aclient, result.image_nodes, self.project.id
        )
        return sorted(
            node_w_scores + image_nodes_w_scores, key=lambda x: x.score, reverse=True
        )
