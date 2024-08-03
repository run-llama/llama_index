from typing import Any, List, Optional

from llama_cloud import TextNodeWithScore
from llama_cloud.resources.pipelines.client import OMIT, PipelineType

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.ingestion.api_utils import get_aclient, get_client
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import MetadataFilters


class LlamaCloudRetriever(BaseRetriever):
    def __init__(
        self,
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        dense_similarity_top_k: Optional[int] = None,
        sparse_similarity_top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        alpha: Optional[float] = None,
        filters: Optional[MetadataFilters] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        retrieval_mode: Optional[str] = None,
        files_top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Retriever."""
        self.name = name
        self.project_name = project_name
        self._client = get_client(api_key, base_url, app_url, timeout)
        self._aclient = get_aclient(api_key, base_url, app_url, timeout)

        projects = self._client.projects.list_projects(
            project_name=project_name, organization_id=organization_id
        )
        if len(projects) == 0:
            raise ValueError(f"No project found with name {project_name}")

        self.project_id = projects[0].id

        self._dense_similarity_top_k = (
            dense_similarity_top_k if dense_similarity_top_k is not None else OMIT
        )
        self._sparse_similarity_top_k = (
            sparse_similarity_top_k if sparse_similarity_top_k is not None else OMIT
        )
        self._enable_reranking = (
            enable_reranking if enable_reranking is not None else OMIT
        )
        self._rerank_top_n = rerank_top_n if rerank_top_n is not None else OMIT
        self._alpha = alpha if alpha is not None else OMIT
        self._filters = filters if filters is not None else OMIT
        self._retrieval_mode = retrieval_mode if retrieval_mode is not None else OMIT
        self._files_top_k = files_top_k if files_top_k is not None else OMIT

        super().__init__(
            callback_manager=kwargs.get("callback_manager", None),
            verbose=kwargs.get("verbose", False),
        )

    def _result_nodes_to_node_with_score(
        self, result_nodes: List[TextNodeWithScore]
    ) -> List[NodeWithScore]:
        nodes = []
        for res in result_nodes:
            text_node = TextNode.parse_obj(res.node.dict())
            nodes.append(NodeWithScore(node=text_node, score=res.score))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from the platform."""
        pipelines = self._client.pipelines.search_pipelines(
            project_name=self.project_name,
            project_id=self.project_id,
            pipeline_name=self.name,
            pipeline_type=PipelineType.MANAGED.value,
        )
        if len(pipelines) == 0:
            raise ValueError(
                f"Unknown index name {self.name}. Please confirm a "
                "managed index with this name exists."
            )
        elif len(pipelines) > 1:
            raise ValueError(
                f"Multiple pipelines found with name {self.name} in project {self.project_name}"
            )
        pipeline = pipelines[0]

        if pipeline.id is None:
            raise ValueError(
                f"No pipeline found with name {self.name} in project {self.project_name}"
            )

        results = self._client.pipelines.run_search(
            query=query_bundle.query_str,
            pipeline_id=pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k,
            sparse_similarity_top_k=self._sparse_similarity_top_k,
            enable_reranking=self._enable_reranking,
            rerank_top_n=self._rerank_top_n,
            alpha=self._alpha,
            search_filters=self._filters,
            files_top_k=self._files_top_k,
            retrieval_mode=self._retrieval_mode,
        )

        result_nodes = results.retrieval_nodes

        return self._result_nodes_to_node_with_score(result_nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        pipelines = await self._aclient.pipelines.search_pipelines(
            project_name=self.project_name,
            pipeline_name=self.name,
            pipeline_type=PipelineType.MANAGED.value,
            project_id=self.project_id,
        )
        if len(pipelines) == 0:
            raise ValueError(
                f"Unknown index name {self.name}. Please confirm a "
                "managed index with this name exists."
            )
        elif len(pipelines) > 1:
            raise ValueError(
                f"Multiple pipelines found with name {self.name} in project {self.project_name}"
            )
        pipeline = pipelines[0]

        if pipeline.id is None:
            raise ValueError(
                f"No pipeline found with name {self.name} in project {self.project_name}"
            )

        results = await self._aclient.pipelines.run_search(
            query=query_bundle.query_str,
            pipeline_id=pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k,
            sparse_similarity_top_k=self._sparse_similarity_top_k,
            enable_reranking=self._enable_reranking,
            rerank_top_n=self._rerank_top_n,
            alpha=self._alpha,
            search_filters=self._filters,
            files_top_k=self._files_top_k,
            retrieval_mode=self._retrieval_mode,
        )

        result_nodes = results.retrieval_nodes

        return self._result_nodes_to_node_with_score(result_nodes)
