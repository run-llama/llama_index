"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import httpx
import os
import time
from typing import Any, List, Optional, Sequence, Type
from urllib.parse import quote_plus

from llama_cloud import (
    PipelineType,
    ProjectCreate,
    ManagedIngestionStatus,
    CloudDocumentCreate,
    CloudDocument,
)

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_APP_URL, DEFAULT_PROJECT_NAME
from llama_index.core.indices.managed.base import BaseManagedIndex

from llama_cloud.core.api_error import ApiError
from llama_index.core.ingestion.api_utils import (
    get_aclient,
    get_client,
)
from llama_index.indices.managed.llama_cloud.api_utils import (
    default_transformations,
    get_pipeline_create,
)
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.settings import Settings
from typing import Any, Dict, List, Optional, Sequence, Type

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.settings import (
    Settings,
)
from llama_index.core.storage.docstore.types import RefDocInfo
import logging

logger = logging.getLogger(__name__)


class LlamaCloudIndex(BaseManagedIndex):
    """LlamaIndex Platform Index."""

    def __init__(
        self,
        name: str,
        nodes: Optional[List[BaseNode]] = None,
        transformations: Optional[List[TransformComponent]] = None,
        timeout: int = 60,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        httpx_client: Optional[httpx.Client] = None,
        async_httpx_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Index."""
        self.name = name
        self.project_name = project_name
        self.organization_id = organization_id
        self.transformations = transformations or []

        if nodes is not None:
            # TODO: How to handle uploading nodes without running transforms on them?
            raise ValueError("LlamaCloudIndex does not support nodes on initialization")

        self._httpx_client = httpx_client
        self._async_httpx_client = async_httpx_client
        self._client = get_client(api_key, base_url, app_url, timeout, httpx_client)
        self._aclient = get_aclient(
            api_key, base_url, app_url, timeout, async_httpx_client
        )

        self._api_key = api_key
        self._base_url = base_url
        self._app_url = app_url
        self._timeout = timeout
        self._show_progress = show_progress
        self._service_context = None
        self._callback_manager = callback_manager or Settings.callback_manager

    def _wait_for_pipeline_ingestion(
        self,
        verbose: bool = False,
        raise_on_partial_success: bool = False,
    ) -> None:
        pipeline_id = self._get_pipeline_id()
        client = self._client

        if verbose:
            print("Syncing pipeline: ", end="")

        is_done = False
        while not is_done:
            status = client.pipelines.get_pipeline_status(
                pipeline_id=pipeline_id
            ).status
            if status == ManagedIngestionStatus.ERROR or (
                raise_on_partial_success
                and status == ManagedIngestionStatus.PARTIAL_SUCCESS
            ):
                raise ValueError(f"Pipeline ingestion failed for {pipeline_id}")
            elif status in [
                ManagedIngestionStatus.NOT_STARTED,
                ManagedIngestionStatus.IN_PROGRESS,
            ]:
                if verbose:
                    print(".", end="")
                time.sleep(0.5)
            else:
                is_done = True
                if verbose:
                    print("Done!")

    def _wait_for_documents_ingestion(
        self,
        doc_ids: List[str],
        verbose: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        pipeline_id = self._get_pipeline_id()
        client = self._client
        if verbose:
            print("Loading data: ", end="")

        # wait until all documents are loaded
        pending_docs = set(doc_ids)
        while pending_docs:
            docs_to_remove = set()
            for doc in pending_docs:
                # we have to quote the doc id twice because it is used as a path parameter
                status = client.pipelines.get_pipeline_document_status(
                    pipeline_id=pipeline_id, document_id=quote_plus(quote_plus(doc))
                )
                if status in [
                    ManagedIngestionStatus.NOT_STARTED,
                    ManagedIngestionStatus.IN_PROGRESS,
                ]:
                    continue

                if status == ManagedIngestionStatus.ERROR:
                    if verbose:
                        print(f"Document ingestion failed for {doc}")
                    if raise_on_error:
                        raise ValueError(f"Document ingestion failed for {doc}")

                docs_to_remove.add(doc)

            pending_docs -= docs_to_remove

            if pending_docs:
                if verbose:
                    print(".", end="")
                time.sleep(0.5)

        if verbose:
            print("Done!")

        # we have to wait for pipeline ingestion because retrieval only works when
        # the pipeline status is success
        self._wait_for_pipeline_ingestion(verbose, raise_on_error)

    def _get_project_id(self) -> str:
        projects = self._client.projects.list_projects(
            organization_id=self.organization_id,
            project_name=self.project_name,
        )
        if len(projects) == 0:
            raise ValueError(
                f"Unknown project name {self.project_name}. Please confirm a "
                "managed project with this name exists."
            )
        elif len(projects) > 1:
            raise ValueError(
                f"Multiple projects found with name {self.project_name}. Please specify organization_id."
            )
        project = projects[0]

        if project.id is None:
            raise ValueError(f"No project found with name {self.project_name}")

        return project.id

    def _get_pipeline_id(self) -> str:
        project_id = self._get_project_id()
        pipelines = self._client.pipelines.search_pipelines(
            project_id=project_id,
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

        return pipeline.id

    @classmethod
    def from_documents(  # type: ignore
        cls: Type["LlamaCloudIndex"],
        documents: List[Document],
        name: str,
        transformations: Optional[List[TransformComponent]] = None,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        verbose: bool = False,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> "LlamaCloudIndex":
        """Build a LlamaCloud managed index from a sequence of documents."""
        app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
        client = get_client(api_key, base_url, app_url, timeout)

        pipeline_create = get_pipeline_create(
            name,
            client,
            PipelineType.MANAGED,
            project_name=project_name,
            transformations=transformations or default_transformations(),
            input_nodes=documents,
        )

        project = client.projects.upsert_project(
            organization_id=organization_id, request=ProjectCreate(name=project_name)
        )
        if project.id is None:
            raise ValueError(f"Failed to create/get project {project_name}")
        if verbose:
            print(f"Created project {project.id} with name {project.name}")

        pipeline = client.pipelines.upsert_pipeline(
            project_id=project.id, request=pipeline_create
        )
        if pipeline.id is None:
            raise ValueError(f"Failed to create/get pipeline {name}")
        if verbose:
            print(f"Created pipeline {pipeline.id} with name {pipeline.name}")

        index = cls(
            name,
            transformations=transformations,
            project_name=project_name,
            organization_id=project.organization_id,
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            **kwargs,
        )

        # this kicks off document ingestion
        upserted_documents = client.pipelines.upsert_batch_pipeline_documents(
            pipeline_id=pipeline.id,
            request=[
                CloudDocumentCreate(
                    text=doc.text,
                    metadata=doc.metadata,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                    id=doc.id_,
                )
                for doc in documents
            ],
        )
        doc_ids = [doc.id for doc in upserted_documents]
        index._wait_for_documents_ingestion(
            doc_ids, verbose=verbose, raise_on_error=raise_on_error
        )

        print(f"Find your index at {app_url}/project/{project.id}/deploy/{pipeline.id}")

        return index

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.llama_cloud.retriever import (
            LlamaCloudRetriever,
        )

        similarity_top_k = kwargs.pop("similarity_top_k", None)
        dense_similarity_top_k = kwargs.pop("dense_similarity_top_k", None)
        if similarity_top_k is not None:
            dense_similarity_top_k = similarity_top_k

        return LlamaCloudRetriever(
            self.name,
            project_name=self.project_name,
            api_key=self._api_key,
            base_url=self._base_url,
            app_url=self._app_url,
            timeout=self._timeout,
            organization_id=self.organization_id,
            dense_similarity_top_k=dense_similarity_top_k,
            httpx_client=self._httpx_client,
            async_httpx_client=self._async_httpx_client,
            **kwargs,
        )

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        kwargs["retriever"] = self.as_retriever(**kwargs)
        return RetrieverQueryEngine.from_args(**kwargs)

    @property
    def ref_doc_info(self, batch_size: int = 100) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their metadata. The nodes list is empty."""
        pipeline_id = self._get_pipeline_id()
        pipeline_documents: List[CloudDocument] = []
        skip = 0
        limit = batch_size
        while True:
            batch = self._client.pipelines.list_pipeline_documents(
                pipeline_id=pipeline_id,
                skip=skip,
                limit=limit,
            )
            if not batch:
                break
            pipeline_documents.extend(batch)
            skip += limit
        return {
            doc.id: RefDocInfo(metadata=doc.metadata, node_ids=[])
            for doc in pipeline_documents
        }

    def insert(
        self, document: Document, verbose: bool = False, **insert_kwargs: Any
    ) -> None:
        """Insert a document."""
        with self._callback_manager.as_trace("insert"):
            pipeline_id = self._get_pipeline_id()
            upserted_documents = self._client.pipelines.create_batch_pipeline_documents(
                pipeline_id=pipeline_id,
                request=[
                    CloudDocumentCreate(
                        text=document.text,
                        metadata=document.metadata,
                        excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                        id=document.id_,
                    )
                ],
            )
            upserted_document = upserted_documents[0]
            self._wait_for_documents_ingestion(
                [upserted_document.id], verbose=verbose, raise_on_error=True
            )

    def update_ref_doc(
        self, document: Document, verbose: bool = False, **update_kwargs: Any
    ) -> None:
        """Upserts a document and its corresponding nodes."""
        with self._callback_manager.as_trace("update"):
            pipeline_id = self._get_pipeline_id()
            upserted_documents = self._client.pipelines.upsert_batch_pipeline_documents(
                pipeline_id=pipeline_id,
                request=[
                    CloudDocumentCreate(
                        text=document.text,
                        metadata=document.metadata,
                        excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                        id=document.id_,
                    )
                ],
            )
            upserted_document = upserted_documents[0]
            self._wait_for_documents_ingestion(
                [upserted_document.id], verbose=verbose, raise_on_error=True
            )

    def refresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed."""
        with self._callback_manager.as_trace("refresh"):
            pipeline_id = self._get_pipeline_id()
            upserted_documents = self._client.pipelines.upsert_batch_pipeline_documents(
                pipeline_id=pipeline_id,
                request=[
                    CloudDocumentCreate(
                        text=doc.text,
                        metadata=doc.metadata,
                        excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                        id=doc.id_,
                    )
                    for doc in documents
                ],
            )
            doc_ids = [doc.id for doc in upserted_documents]
            self._wait_for_documents_ingestion(
                doc_ids, verbose=True, raise_on_error=True
            )
            return [True] * len(doc_ids)

    def delete_ref_doc(
        self,
        ref_doc_id: str,
        delete_from_docstore: bool = False,
        verbose: bool = False,
        raise_if_not_found: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a document and its nodes by using ref_doc_id."""
        pipeline_id = self._get_pipeline_id()
        try:
            # we have to quote the ref_doc_id twice because it is used as a path parameter
            self._client.pipelines.delete_pipeline_document(
                pipeline_id=pipeline_id, document_id=quote_plus(quote_plus(ref_doc_id))
            )
        except ApiError as e:
            if e.status_code == 404 and not raise_if_not_found:
                logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            else:
                raise

        # we have to wait for the pipeline instead of the document, because the document is already deleted
        self._wait_for_pipeline_ingestion(
            verbose=verbose, raise_on_partial_success=False
        )

    # Nodes related methods (not implemented for LlamaCloudIndex)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        raise NotImplementedError("_insert not implemented for LlamaCloudIndex.")

    def build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """Build the index from nodes."""
        raise NotImplementedError(
            "build_index_from_nodes not implemented for LlamaCloudIndex."
        )

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of nodes."""
        raise NotImplementedError("insert_nodes not implemented for LlamaCloudIndex.")

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a set of nodes."""
        raise NotImplementedError("delete_nodes not implemented for LlamaCloudIndex.")
