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
    ManagedIngestionStatusResponse,
    PipelineCreate,
    PipelineCreateEmbeddingConfig,
    PipelineCreateTransformConfig,
    PipelineType,
    ProjectCreate,
    ManagedIngestionStatus,
    CloudDocumentCreate,
    CloudDocument,
    PipelineFileCreate,
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
from llama_index.indices.managed.llama_cloud.api_utils import (
    default_embedding_config,
    default_transform_config,
    resolve_project_and_pipeline,
)
import logging

logger = logging.getLogger(__name__)


class LlamaCloudIndex(BaseManagedIndex):
    """
    A managed index that stores documents in LlamaCloud.

    There are two main ways to use this index:

    1. Connect to an existing LlamaCloud index:
        ```python
        # Connect using index ID (same as pipeline ID)
        index = LlamaCloudIndex(id="<index_id>")

        # Or connect using index name
        index = LlamaCloudIndex(
            name="my_index",
            project_name="my_project",
            organization_id="my_org_id"
        )
        ```

    2. Create a new index with documents:
        ```python
        documents = [Document(...), Document(...)]
        index = LlamaCloudIndex.from_documents(
            documents,
            name="my_new_index",
            project_name="my_project",
            organization_id="my_org_id"
        )
        ```

    The index supports standard operations like retrieval and querying
    through the as_query_engine() and as_retriever() methods.
    """

    def __init__(
        self,
        # index identifier
        name: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        index_id: Optional[str] = None,  # alias for pipeline_id
        id: Optional[str] = None,  # alias for pipeline_id
        # project identifier
        project_id: Optional[str] = None,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        # connection params
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        httpx_client: Optional[httpx.Client] = None,
        async_httpx_client: Optional[httpx.AsyncClient] = None,
        # misc
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        # deprecated
        nodes: Optional[List[BaseNode]] = None,
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Index."""
        if sum([bool(id), bool(index_id), bool(pipeline_id), bool(name)]) != 1:
            raise ValueError(
                "Exactly one of `name`, `id`, `pipeline_id` or `index_id` must be provided to identify the index."
            )

        if nodes is not None:
            # TODO: How to handle uploading nodes without running transforms on them?
            raise ValueError("LlamaCloudIndex does not support nodes on initialization")

        if transformations is not None:
            raise ValueError(
                "Setting transformations is deprecated for LlamaCloudIndex, please use the `transform_config` and `embedding_config` parameters instead."
            )

        # initialize clients
        self._httpx_client = httpx_client
        self._async_httpx_client = async_httpx_client
        self._client = get_client(api_key, base_url, app_url, timeout, httpx_client)
        self._aclient = get_aclient(
            api_key, base_url, app_url, timeout, async_httpx_client
        )

        self.organization_id = organization_id
        pipeline_id = id or index_id or pipeline_id

        self.project, self.pipeline = resolve_project_and_pipeline(
            self._client, name, pipeline_id, project_name, project_id, organization_id
        )
        self.name = self.pipeline.name
        self.project_name = self.project.name

        self._api_key = api_key
        self._base_url = base_url
        self._app_url = app_url
        self._timeout = timeout
        self._show_progress = show_progress
        self._service_context = None
        self._callback_manager = callback_manager or Settings.callback_manager

    @property
    def id(self) -> str:
        """Return the pipeline (aka index) ID."""
        return self.pipeline.id

    def wait_for_completion(
        self,
        verbose: bool = False,
        raise_on_partial_success: bool = False,
        sleep_interval: float = 0.5,
    ) -> Optional[ManagedIngestionStatusResponse]:
        if sleep_interval < 0.5:
            # minimum sleep interval at 0.5 seconds to prevent rate-limiting
            sleep_interval = 0.5
        if verbose:
            print(f"Syncing pipeline {self.pipeline.id}: ", end="")

        is_done = False
        status_response: Optional[ManagedIngestionStatusResponse] = None
        while not is_done:
            status_response = self._client.pipelines.get_pipeline_status(
                pipeline_id=self.pipeline.id
            )
            status = status_response.status
            if status == ManagedIngestionStatus.ERROR or (
                raise_on_partial_success
                and status == ManagedIngestionStatus.PARTIAL_SUCCESS
            ):
                error_details = status_response.json()
                raise ValueError(
                    f"Pipeline ingestion failed for {self.pipeline.id}. Error details: {error_details}"
                )
            elif status in [
                ManagedIngestionStatus.NOT_STARTED,
                ManagedIngestionStatus.IN_PROGRESS,
            ]:
                if verbose:
                    print(".", end="")
                time.sleep(sleep_interval)
            else:
                is_done = True
                if verbose:
                    print("Done!")
        return status_response

    def _wait_for_file_ingestion(
        self,
        file_id: str,
        verbose: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        if verbose:
            print("Loading file: ", end="")

        # wait until the file is loaded
        is_done = False
        while not is_done:
            status = self._client.pipelines.get_pipeline_file_status(
                pipeline_id=self.pipeline.id, file_id=file_id
            ).status
            if status == ManagedIngestionStatus.ERROR:
                if verbose:
                    print(f"File ingestion failed for {file_id}")
                if raise_on_error:
                    raise ValueError(f"File ingestion failed for {file_id}")
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
        if verbose:
            print("Loading data: ", end="")

        # wait until all documents are loaded
        pending_docs = set(doc_ids)
        while pending_docs:
            docs_to_remove = set()
            for doc in pending_docs:
                # we have to quote the doc id twice because it is used as a path parameter
                status = self._client.pipelines.get_pipeline_document_status(
                    pipeline_id=self.pipeline.id,
                    document_id=quote_plus(quote_plus(doc)),
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
        self.wait_for_completion(verbose, raise_on_error)

    @classmethod
    def from_documents(  # type: ignore
        cls: Type["LlamaCloudIndex"],
        documents: List[Document],
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        verbose: bool = False,
        raise_on_error: bool = False,
        # ingestion configs
        embedding_config: Optional[PipelineCreateEmbeddingConfig] = None,
        transform_config: Optional[PipelineCreateTransformConfig] = None,
        # deprecated
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any,
    ) -> "LlamaCloudIndex":
        """Build a LlamaCloud managed index from a sequence of documents."""
        app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
        client = get_client(api_key, base_url, app_url, timeout)

        if transformations is not None:
            raise ValueError(
                "Setting transformations is deprecated for LlamaCloudIndex"
            )

        # create project if it doesn't exist
        project = client.projects.upsert_project(
            organization_id=organization_id, request=ProjectCreate(name=project_name)
        )
        if project.id is None:
            raise ValueError(f"Failed to create/get project {project_name}")
        if verbose:
            print(f"Created project {project.id} with name {project.name}")

        # create pipeline
        pipeline_create = PipelineCreate(
            name=name,
            pipeline_type=PipelineType.MANAGED,
            embedding_config=embedding_config or default_embedding_config(),
            transform_config=transform_config or default_transform_config(),
            # we are uploading document directly, so we don't need llama parse
            llama_parse_enabled=False,
        )
        pipeline = client.pipelines.upsert_pipeline(
            project_id=project.id, request=pipeline_create
        )
        if pipeline.id is None:
            raise ValueError(f"Failed to create/get pipeline {name}")
        if verbose:
            print(f"Created pipeline {pipeline.id} with name {pipeline.name}")

        index = cls(
            name,
            project_name=project.name,
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
            project_id=self.project.id,
            pipeline_id=self.pipeline.id,
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
        pipeline_id = self.pipeline.id
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
            upserted_documents = self._client.pipelines.create_batch_pipeline_documents(
                pipeline_id=self.pipeline.id,
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
            upserted_documents = self._client.pipelines.upsert_batch_pipeline_documents(
                pipeline_id=self.pipeline.id,
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
            upserted_documents = self._client.pipelines.upsert_batch_pipeline_documents(
                pipeline_id=self.pipeline.id,
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
        try:
            # we have to quote the ref_doc_id twice because it is used as a path parameter
            self._client.pipelines.delete_pipeline_document(
                pipeline_id=self.pipeline.id,
                document_id=quote_plus(quote_plus(ref_doc_id)),
            )
        except ApiError as e:
            if e.status_code == 404 and not raise_if_not_found:
                logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            else:
                raise

        # we have to wait for the pipeline instead of the document, because the document is already deleted
        self.wait_for_completion(verbose=verbose, raise_on_partial_success=False)

    def upload_file(
        self,
        file_path: str,
        resource_info: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        wait_for_ingestion: bool = True,
        raise_on_error: bool = False,
    ) -> str:
        """Upload a file to the index."""
        with open(file_path, "rb") as f:
            file = self._client.files.upload_file(
                project_id=self.project.id, upload_file=f
            )
            if verbose:
                print(f"Uploaded file {file.id} with name {file.name}")
        if resource_info:
            self._client.files.update(file_id=file.id, request=resource_info)
        # Add file to pipeline
        pipeline_file_create = PipelineFileCreate(file_id=file.id)
        self._client.pipelines.add_files_to_pipeline(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            self._wait_for_file_ingestion(
                file.id, verbose=verbose, raise_on_error=raise_on_error
            )
        return file.id

    def upload_file_from_url(
        self,
        file_name: str,
        url: str,
        proxy_url: Optional[str] = None,
        request_headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        follow_redirects: bool = True,
        verbose: bool = False,
        wait_for_ingestion: bool = True,
        raise_on_error: bool = False,
    ) -> str:
        """Upload a file from a URL to the index."""
        file = self._client.files.upload_file_from_url(
            project_id=self.project.id,
            name=file_name,
            url=url,
            proxy_url=proxy_url,
            request_headers=request_headers,
            verify_ssl=verify_ssl,
            follow_redirects=follow_redirects,
        )
        if verbose:
            print(f"Uploaded file {file.id} with ID {file.id}")

        # Add file to pipeline
        pipeline_file_create = PipelineFileCreate(file_id=file.id)
        self._client.pipelines.add_files_to_pipeline(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            self._wait_for_file_ingestion(
                file.id, verbose=verbose, raise_on_error=raise_on_error
            )
        return file.id

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
