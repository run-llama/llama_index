"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""
import time
from typing import Any, List, Optional, Sequence, Type

from llama_index_client import PipelineType, ProjectCreate

from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.llamaindex.utils import (
    default_transformations,
    get_aclient,
    get_client,
    get_pipeline_create,
)
from llama_index.ingestion.pipeline import DEFAULT_PROJECT_NAME
from llama_index.schema import BaseNode, Document, TransformComponent
from llama_index.service_context import ServiceContext


class PlatformIndex(BaseManagedIndex):
    """Platform Index.

    TODO: Docstring

    """

    def __init__(
        self,
        name: str,
        nodes: Optional[List[BaseNode]] = None,
        transformations: Optional[List[TransformComponent]] = None,
        timeout: int = 60,
        project_name: str = DEFAULT_PROJECT_NAME,
        platform_api_key: Optional[str] = None,
        platform_base_url: Optional[str] = None,
        platform_app_url: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Index."""
        self.name = name
        self.project_name = project_name
        self.transformations = transformations or []

        if not service_context and len(self.transformations) == 0:
            self.transformations = default_transformations()
        elif service_context and len(self.transformations) == 0:
            self.transformations = service_context.transformations

        if nodes is not None:
            # TODO: How to handle uploading nodes without running transforms on them?
            raise ValueError("PlatformIndex does not support nodes on initialization")

        self._client = get_client(
            platform_api_key, platform_base_url, platform_app_url, timeout
        )
        self._aclient = get_aclient(
            platform_api_key, platform_base_url, platform_app_url, timeout
        )

        self._platform_api_key = platform_api_key
        self._platform_base_url = platform_base_url
        self._platform_app_url = platform_app_url
        self._timeout = timeout
        self._show_progress = show_progress
        self._service_context = service_context  # type: ignore

    @classmethod
    def from_documents(  # type: ignore
        cls: Type["PlatformIndex"],
        name: str,
        documents: List[Document],
        transformations: Optional[List[TransformComponent]] = None,
        project_name: str = DEFAULT_PROJECT_NAME,
        platform_api_key: Optional[str] = None,
        platform_base_url: Optional[str] = None,
        platform_app_url: Optional[str] = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> "PlatformIndex":
        """Build a Vectara index from a sequence of documents."""
        client = get_client(
            platform_api_key, platform_base_url, platform_app_url, timeout
        )

        pipeline_create = get_pipeline_create(
            name,
            client,
            PipelineType.MANAGED,
            project_name=project_name,
            transformations=transformations or default_transformations(),
            input_nodes=documents,
        )

        project = client.project.upsert_project(
            request=ProjectCreate(name=project_name)
        )
        assert project.id is not None

        pipeline = client.project.upsert_pipeline_for_project(
            project_id=project.id, request=pipeline_create
        )
        assert pipeline.id is not None

        # TODO: remove when sourabh's PR is merged
        for data_source in pipeline.data_sources:
            client.data_source.create_data_source_execution(
                data_source_id=data_source.id
            )
            time.sleep(120)

        # kick off execution
        execution = client.pipeline.run_managed_pipeline_ingestion(
            pipeline_id=pipeline.id
        )
        assert execution.id is not None
        print(execution.id)

        # TODO: Update when pipeline status is available
        # assert execution.status is not None

        # execution = client.pipeline.get_managed_ingestion_execution(
        #     pipline_id=pipeline.id, managed_pipeline_ingestion_id=execution.id
        # )

        # while execution.status not in ["SUCCEEDED", "FAILED"]:
        #     time.sleep(1)
        #     execution = client.pipeline.get_managed_ingestion_execution(
        #         pipline_id=pipeline.id, managed_pipeline_ingestion_id=execution.id
        #     )
        #     assert execution.status is not None

        # TODO: What is the actual URL?
        print(
            f"Find your deployed pipeline at {platform_app_url}/pipelines/{pipeline.id}"
        )

        return cls(
            name,
            transformations=transformations,
            project_name=project_name,
            platform_api_key=platform_api_key,
            platform_base_url=platform_base_url,
            platform_app_url=platform_app_url,
            timeout=timeout,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.llamaindex.retriever import PlatformRetriever

        similarity_top_k = kwargs.get("similarity_top_k", None)
        dense_similarity_top_k = kwargs.get("dense_similarity_top_k", None)
        if similarity_top_k is not None:
            dense_similarity_top_k = similarity_top_k

        return PlatformRetriever(
            self.name,
            project_name=self.project_name,
            platform_api_key=self._platform_api_key,
            platform_base_url=self._platform_base_url,
            platform_app_url=self._platform_app_url,
            timeout=self._timeout,
            dense_similarity_top_k=dense_similarity_top_k,
            **kwargs,
        )

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        kwargs["retriever"] = self.as_retriever(**kwargs)
        return RetrieverQueryEngine.from_args(**kwargs)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of documents (each a node)."""
        raise NotImplementedError("_insert not implemented for PlatformIndex.")

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        raise NotImplementedError("delete_ref_doc not implemented for PlatformIndex.")

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes."""
        raise NotImplementedError("update_ref_doc not implemented for PlatformIndex.")
