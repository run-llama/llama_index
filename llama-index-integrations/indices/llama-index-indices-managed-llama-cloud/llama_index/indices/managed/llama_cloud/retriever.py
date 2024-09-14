from typing import Any, List, Optional

from llama_cloud import TextNodeWithScore, PageScreenshotNodeWithScore
from llama_cloud.resources.pipelines.client import OMIT, PipelineType
from llama_cloud.client import LlamaCloud, AsyncLlamaCloud
from llama_cloud.core import remove_none_from_dict
from llama_cloud.core.api_error import ApiError

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.constants import DEFAULT_PROJECT_NAME
from llama_index.core.ingestion.api_utils import get_aclient, get_client
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, ImageNode
from llama_index.core.vector_stores.types import MetadataFilters
import asyncio
import urllib.parse
import base64


def _get_page_screenshot(
    client: LlamaCloud, file_id: str, page_index: int, project_id: str
) -> str:
    """Get the page screenshot."""
    # TODO: this currently uses requests, should be replaced with the client
    _response = client._client_wrapper.httpx_client.request(
        "GET",
        urllib.parse.urljoin(
            f"{client._client_wrapper.get_base_url()}/",
            f"api/v1/files/{file_id}/page_screenshots/{page_index}",
        ),
        params=remove_none_from_dict({"project_id": project_id}),
        headers=client._client_wrapper.get_headers(),
        timeout=60,
    )
    if 200 <= _response.status_code < 300:
        return _response.content
    else:
        raise ApiError(status_code=_response.status_code, body=_response.text)


async def _aget_page_screenshot(
    client: AsyncLlamaCloud, file_id: str, page_index: int, project_id: str
) -> str:
    """Get the page screenshot."""
    # TODO: this currently uses requests, should be replaced with the client
    _response = await client._client_wrapper.httpx_client.request(
        "GET",
        urllib.parse.urljoin(
            f"{client._client_wrapper.get_base_url()}/",
            f"api/v1/files/{file_id}/page_screenshots/{page_index}",
        ),
        params=remove_none_from_dict({"project_id": project_id}),
        headers=client._client_wrapper.get_headers(),
        timeout=60,
    )
    if 200 <= _response.status_code < 300:
        return _response.content
    else:
        raise ApiError(status_code=_response.status_code, body=_response.text)


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
        retrieve_image_nodes: Optional[bool] = None,
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
        self._retrieve_image_nodes = (
            retrieve_image_nodes if retrieve_image_nodes is not None else OMIT
        )

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

    def _image_nodes_to_node_with_score(
        self, raw_image_nodes: List[PageScreenshotNodeWithScore]
    ) -> List[NodeWithScore]:
        image_nodes = []
        if self._retrieve_image_nodes:
            for raw_image_node in raw_image_nodes:
                # TODO: this is a hack to use requests, should be replaced with the client
                image_bytes = _get_page_screenshot(
                    client=self._client,
                    file_id=raw_image_node.node.file_id,
                    page_index=raw_image_node.node.page_index,
                    project_id=self.project_id,
                )
                # Convert image bytes to base64 encoded string
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_node_with_score = NodeWithScore(
                    node=ImageNode(image=image_base64), score=raw_image_node.score
                )
                image_nodes.append(image_node_with_score)
        else:
            if len(image_nodes) > 0:
                raise ValueError(
                    "Image nodes were retrieved but `retrieve_image_nodes` was set to False."
                )
        return image_nodes

    async def _aimage_nodes_to_node_with_score(
        self, raw_image_nodes: List[PageScreenshotNodeWithScore]
    ) -> List[NodeWithScore]:
        image_nodes = []
        if self._retrieve_image_nodes:
            tasks = [
                _aget_page_screenshot(
                    client=self._aclient,
                    file_id=raw_image_node.node.file_id,
                    page_index=raw_image_node.node.page_index,
                    project_id=self.project_id,
                )
                for raw_image_node in raw_image_nodes
            ]

            image_bytes_list = await asyncio.gather(*tasks)
            for image_bytes, raw_image_node in zip(image_bytes_list, raw_image_nodes):
                # Convert image bytes to base64 encoded string
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_node_with_score = NodeWithScore(
                    node=ImageNode(image=image_base64), score=raw_image_node.score
                )
                image_nodes.append(image_node_with_score)
        else:
            if len(image_nodes) > 0:
                raise ValueError(
                    "Image nodes were retrieved but `retrieve_image_nodes` was set to False."
                )
        return image_nodes

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
            retrieve_image_nodes=self._retrieve_image_nodes,
        )

        result_nodes = self._result_nodes_to_node_with_score(results.retrieval_nodes)
        result_nodes.extend(self._image_nodes_to_node_with_score(results.image_nodes))

        return result_nodes

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
            retrieve_image_nodes=self._retrieve_image_nodes,
        )

        result_nodes = self._result_nodes_to_node_with_score(results.retrieval_nodes)
        result_nodes.extend(
            await self._aimage_nodes_to_node_with_score(results.image_nodes)
        )
        return result_nodes
