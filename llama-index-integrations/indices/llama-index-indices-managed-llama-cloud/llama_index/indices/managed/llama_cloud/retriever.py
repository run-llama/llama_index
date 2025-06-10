from typing import Any, List, Optional

import httpx
from llama_cloud import (
    TextNodeWithScore,
)
from llama_cloud.resources.pipelines.client import OMIT

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.ingestion.api_utils import get_aclient, get_client
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.indices.managed.llama_cloud.api_utils import (
    resolve_project_and_pipeline,
    page_screenshot_nodes_to_node_with_score,
    page_figure_nodes_to_node_with_score,
    apage_screenshot_nodes_to_node_with_score,
    apage_figure_nodes_to_node_with_score,
)
import logging


logger = logging.getLogger(__name__)


class LlamaCloudRetriever(BaseRetriever):
    def __init__(
        self,
        # index identifier
        name: Optional[str] = None,
        index_id: Optional[str] = None,  # alias for pipeline_id
        id: Optional[str] = None,  # alias for pipeline_id
        pipeline_id: Optional[str] = None,
        # project identifier
        project_name: Optional[str] = DEFAULT_PROJECT_NAME,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        # connection params
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        httpx_client: Optional[httpx.Client] = None,
        async_httpx_client: Optional[httpx.AsyncClient] = None,
        # retrieval params
        dense_similarity_top_k: Optional[int] = None,
        sparse_similarity_top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        alpha: Optional[float] = None,
        filters: Optional[MetadataFilters] = None,
        retrieval_mode: Optional[str] = None,
        files_top_k: Optional[int] = None,
        retrieve_image_nodes: Optional[bool] = None,
        retrieve_page_screenshot_nodes: Optional[bool] = None,
        retrieve_page_figure_nodes: Optional[bool] = None,
        search_filters_inference_schema: Optional[BaseModel] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Retriever."""
        if sum([bool(id), bool(index_id), bool(pipeline_id), bool(name)]) != 1:
            raise ValueError(
                "Exactly one of `name`, `id`, `pipeline_id` or `index_id` must be provided to identify the index."
            )

        # initialize clients
        self._httpx_client = httpx_client
        self._async_httpx_client = async_httpx_client
        self._client = get_client(api_key, base_url, app_url, timeout, httpx_client)
        self._aclient = get_aclient(
            api_key, base_url, app_url, timeout, async_httpx_client
        )

        pipeline_id = id or index_id or pipeline_id
        self.project, self.pipeline = resolve_project_and_pipeline(
            self._client, name, pipeline_id, project_name, project_id, organization_id
        )
        self.name = self.pipeline.name
        self.project_name = self.project.name

        # retrieval params
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
        if retrieve_image_nodes is not None:
            logger.warning(
                "The `retrieve_image_nodes` parameter is deprecated. "
                "Use `retrieve_page_screenshot_nodes` and `retrieve_page_figure_nodes` instead."
            )
        if retrieve_image_nodes:
            if (
                retrieve_page_screenshot_nodes is False
                or retrieve_page_figure_nodes is False
            ):
                raise ValueError(
                    "If `retrieve_image_nodes` is set to True, "
                    "both `retrieve_page_screenshot_nodes` and `retrieve_page_figure_nodes` must also be set to True or omitted."
                )
            retrieve_page_screenshot_nodes = True
            retrieve_page_figure_nodes = True
        self._retrieve_page_screenshot_nodes = (
            retrieve_page_screenshot_nodes
            if retrieve_page_screenshot_nodes is not None
            else OMIT
        )
        self._retrieve_page_figure_nodes = (
            retrieve_page_figure_nodes
            if retrieve_page_figure_nodes is not None
            else OMIT
        )
        self._search_filters_inference_schema = search_filters_inference_schema

        super().__init__(
            callback_manager=kwargs.get("callback_manager"),
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
        search_filters_inference_schema = OMIT
        if self._search_filters_inference_schema is not None:
            search_filters_inference_schema = (
                self._search_filters_inference_schema.model_json_schema()
            )
        results = self._client.pipelines.run_search(
            query=query_bundle.query_str,
            pipeline_id=self.pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k,
            sparse_similarity_top_k=self._sparse_similarity_top_k,
            enable_reranking=self._enable_reranking,
            rerank_top_n=self._rerank_top_n,
            alpha=self._alpha,
            search_filters=self._filters,
            files_top_k=self._files_top_k,
            retrieval_mode=self._retrieval_mode,
            retrieve_page_screenshot_nodes=self._retrieve_page_screenshot_nodes,
            retrieve_page_figure_nodes=self._retrieve_page_figure_nodes,
            search_filters_inference_schema=search_filters_inference_schema,
        )

        result_nodes = self._result_nodes_to_node_with_score(results.retrieval_nodes)
        if self._retrieve_page_screenshot_nodes:
            result_nodes.extend(
                page_screenshot_nodes_to_node_with_score(
                    self._client, results.image_nodes, self.project.id
                )
            )
        if self._retrieve_page_figure_nodes:
            result_nodes.extend(
                page_figure_nodes_to_node_with_score(
                    self._client, results.page_figure_nodes, self.project.id
                )
            )

        return result_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        search_filters_inference_schema = OMIT
        if self._search_filters_inference_schema is not None:
            search_filters_inference_schema = (
                self._search_filters_inference_schema.model_json_schema()
            )
        results = await self._aclient.pipelines.run_search(
            query=query_bundle.query_str,
            pipeline_id=self.pipeline.id,
            dense_similarity_top_k=self._dense_similarity_top_k,
            sparse_similarity_top_k=self._sparse_similarity_top_k,
            enable_reranking=self._enable_reranking,
            rerank_top_n=self._rerank_top_n,
            alpha=self._alpha,
            search_filters=self._filters,
            files_top_k=self._files_top_k,
            retrieval_mode=self._retrieval_mode,
            retrieve_page_screenshot_nodes=self._retrieve_page_screenshot_nodes,
            retrieve_page_figure_nodes=self._retrieve_page_figure_nodes,
            search_filters_inference_schema=search_filters_inference_schema,
        )

        result_nodes = self._result_nodes_to_node_with_score(results.retrieval_nodes)
        if self._retrieve_page_screenshot_nodes:
            result_nodes.extend(
                await apage_screenshot_nodes_to_node_with_score(
                    self._aclient, results.image_nodes, self.project.id
                )
            )
        if self._retrieve_page_figure_nodes:
            result_nodes.extend(
                await apage_figure_nodes_to_node_with_score(
                    self._aclient, results.image_nodes, self.project.id
                )
            )

        return result_nodes
