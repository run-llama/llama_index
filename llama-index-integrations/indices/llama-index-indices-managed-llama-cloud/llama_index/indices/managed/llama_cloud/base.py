"""
Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import asyncio
import httpx
import os
import time
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Type
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
    LlamaParseParameters,
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
        self._client = get_client(
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            httpx_client=httpx_client,
        )
        self._aclient = get_aclient(
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            httpx_client=async_httpx_client,
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
        self._callback_manager = callback_manager or Settings.callback_manager

    @property
    def id(self) -> str:
        """Return the pipeline (aka index) ID."""
        return self.pipeline.id

    def _wait_for_resources(
        self,
        resource_ids: Sequence[str],
        get_status_fn: Callable[[str], ManagedIngestionStatusResponse],
        resource_name: str,
        verbose: bool,
        raise_on_error: bool,
        sleep_interval: float,
    ) -> None:
        """
        Poll `get_status_fn` until every id in `resource_ids` is finished.

        Args:
            resource_ids: Iterable of resource ids to watch.
            get_status_fn: Callable that maps a resource id → ManagedIngestionStatus.
            resource_name: Text used in log / error messages: "file", "document", ….
            verbose: Print a progress bar.
            raise_on_error: Whether to raise on ManagedIngestionStatus.ERROR.
            sleep_interval: Seconds between polls (min 0.5 s to avoid rate-limits).

        """
        if not resource_ids:  # nothing to do
            return

        if verbose:
            print(
                f"Loading {resource_name}{'s' if len(resource_ids) > 1 else ''}",
            )

        pending: set[str] = set(resource_ids)
        while pending:
            finished: set[str] = set()
            for rid in pending:
                try:
                    status_response = get_status_fn(rid)
                    status = status_response.status
                    if status in (
                        ManagedIngestionStatus.NOT_STARTED,
                        ManagedIngestionStatus.IN_PROGRESS,
                    ):
                        continue  # still working

                    if status == ManagedIngestionStatus.ERROR:
                        if verbose:
                            print(
                                f"{resource_name.capitalize()} ingestion failed for {rid}"
                            )
                        if raise_on_error:
                            raise ValueError(
                                f"{resource_name.capitalize()} ingestion failed for {rid}"
                            )

                    finished.add(rid)
                    if verbose:
                        print(
                            f"{resource_name.capitalize()} ingestion finished for {rid}"
                        )

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 500, 502, 503, 504):
                        pass
                    else:
                        raise

            pending -= finished

            if pending:
                time.sleep(sleep_interval)

        if verbose:
            print("Done!")

    async def _await_for_resources(
        self,
        resource_ids: Sequence[str],
        get_status_fn: Callable[[str], Awaitable[ManagedIngestionStatusResponse]],
        resource_name: str,
        verbose: bool,
        raise_on_error: bool,
        sleep_interval: float,
    ) -> None:
        """
        Poll `get_status_fn` until every id in `resource_ids` is finished.

        Args:
            resource_ids: Iterable of resource ids to watch.
            get_status_fn: Callable that maps a resource id → ManagedIngestionStatus.
            resource_name: Text used in log / error messages: "file", "document", ….
            verbose: Print a progress bar.
            raise_on_error: Whether to raise on ManagedIngestionStatus.ERROR.
            sleep_interval: Seconds between polls (min 0.5 s to avoid rate-limits).

        """
        if not resource_ids:  # nothing to do
            return

        if verbose:
            print(
                f"Loading {resource_name}{'s' if len(resource_ids) > 1 else ''}",
            )

        pending: set[str] = set(resource_ids)
        while pending:
            finished: set[str] = set()
            for rid in pending:
                try:
                    status_response = await get_status_fn(rid)
                    status = status_response.status
                    if status in (
                        ManagedIngestionStatus.NOT_STARTED,
                        ManagedIngestionStatus.IN_PROGRESS,
                    ):
                        continue  # still working

                    if status == ManagedIngestionStatus.ERROR:
                        if verbose:
                            print(
                                f"{resource_name.capitalize()} ingestion failed for {rid}"
                            )
                        if raise_on_error:
                            raise ValueError(
                                f"{resource_name.capitalize()} ingestion failed for {rid}"
                            )

                    finished.add(rid)
                    if verbose:
                        print(
                            f"{resource_name.capitalize()} ingestion finished for {rid}"
                        )

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 500, 502, 503, 504):
                        pass
                    else:
                        raise

            pending -= finished

            if pending:
                await asyncio.sleep(sleep_interval)

        if verbose:
            print("Done!")

    def wait_for_completion(
        self,
        file_ids: Optional[Sequence[str]] = None,
        doc_ids: Optional[Sequence[str]] = None,
        verbose: bool = False,
        raise_on_partial_success: bool = False,
        raise_on_error: bool = False,
        sleep_interval: float = 1.0,
    ) -> Optional[ManagedIngestionStatusResponse]:
        """
        Block until the requested ingestion work is finished.

        - If `file_ids` is given → wait for those files.
        - If `doc_ids` is given → wait for those documents.
        - If neither is given → wait for the pipeline itself last so that retrieval works.
        - Always waits for the pipeline itself last so that retrieval works.

        Returns the final PipelineStatus response (or None if only waiting on
        files / documents).
        """
        # Batch of files (if any)
        if file_ids:
            self._wait_for_resources(
                file_ids,
                lambda fid: self._client.pipelines.get_pipeline_file_status(
                    pipeline_id=self.pipeline.id, file_id=fid
                ),
                resource_name="file",
                verbose=verbose,
                raise_on_error=raise_on_error,
                sleep_interval=sleep_interval,
            )

        # Batch of documents (if any)
        if doc_ids:
            self._wait_for_resources(
                doc_ids,
                lambda did: self._client.pipelines.get_pipeline_document_status(
                    pipeline_id=self.pipeline.id,
                    document_id=quote_plus(quote_plus(did)),
                ),
                resource_name="document",
                verbose=verbose,
                raise_on_error=raise_on_error,
                sleep_interval=sleep_interval,
            )

        # Finally, wait for the pipeline
        if verbose:
            print(f"Syncing pipeline {self.pipeline.id}")

        status_response: Optional[ManagedIngestionStatusResponse] = None
        while True:
            try:
                status_response = self._client.pipelines.get_pipeline_status(
                    pipeline_id=self.pipeline.id
                )
                status = status_response.status
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(sleep_interval)
                    continue
                else:
                    raise

            if status == ManagedIngestionStatus.ERROR or (
                raise_on_partial_success
                and status == ManagedIngestionStatus.PARTIAL_SUCCESS
            ):
                raise ValueError(
                    f"Pipeline ingestion failed for {self.pipeline.id}. "
                    f"Details: {status_response.json()}"
                )

            if status in (
                ManagedIngestionStatus.NOT_STARTED,
                ManagedIngestionStatus.IN_PROGRESS,
            ):
                if verbose:
                    print(".", end="")
                time.sleep(sleep_interval)
            else:
                if verbose:
                    print("Done!")

                return status_response

    async def await_for_completion(
        self,
        file_ids: Optional[Sequence[str]] = None,
        doc_ids: Optional[Sequence[str]] = None,
        verbose: bool = False,
        raise_on_partial_success: bool = False,
        raise_on_error: bool = False,
        sleep_interval: float = 1.0,
    ) -> Optional[ManagedIngestionStatusResponse]:
        """
        Block until the requested ingestion work is finished.

        - If `file_ids` is given → wait for those files.
        - If `doc_ids` is given → wait for those documents.
        - If neither is given → wait for the pipeline itself last so that retrieval works.
        - Always waits for the pipeline itself last so that retrieval works.

        Returns the final PipelineStatus response (or None if only waiting on
        files / documents).
        """
        # Batch of files (if any)
        if file_ids:
            await self._await_for_resources(
                file_ids,
                lambda fid: self._aclient.pipelines.get_pipeline_file_status(
                    pipeline_id=self.pipeline.id, file_id=fid
                ),
                resource_name="file",
                verbose=verbose,
                raise_on_error=raise_on_error,
                sleep_interval=sleep_interval,
            )

        # Batch of documents (if any)
        if doc_ids:
            await self._await_for_resources(
                doc_ids,
                lambda did: self._aclient.pipelines.get_pipeline_document_status(
                    pipeline_id=self.pipeline.id,
                    document_id=quote_plus(quote_plus(did)),
                ),
                resource_name="document",
                verbose=verbose,
                raise_on_error=raise_on_error,
                sleep_interval=sleep_interval,
            )

        # Finally, wait for the pipeline
        if verbose:
            print(f"Syncing pipeline {self.pipeline.id}")

        status_response: Optional[ManagedIngestionStatusResponse] = None
        while True:
            try:
                status_response = await self._aclient.pipelines.get_pipeline_status(
                    pipeline_id=self.pipeline.id
                )
                status = status_response.status
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(sleep_interval)
                    continue
                else:
                    raise

            if status == ManagedIngestionStatus.ERROR or (
                raise_on_partial_success
                and status == ManagedIngestionStatus.PARTIAL_SUCCESS
            ):
                raise ValueError(
                    f"Pipeline ingestion failed for {self.pipeline.id}. "
                    f"Details: {status_response.json()}"
                )

            if status in (
                ManagedIngestionStatus.NOT_STARTED,
                ManagedIngestionStatus.IN_PROGRESS,
            ):
                if verbose:
                    print(".", end="")
                await asyncio.sleep(sleep_interval)
            else:
                if verbose:
                    print("Done!")

                return status_response

    @classmethod
    def create_index(
        cls: Type["LlamaCloudIndex"],
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        verbose: bool = False,
        # ingestion configs
        embedding_config: Optional[PipelineCreateEmbeddingConfig] = None,
        transform_config: Optional[PipelineCreateTransformConfig] = None,
        llama_parse_parameters: Optional[LlamaParseParameters] = None,
        **kwargs: Any,
    ) -> "LlamaCloudIndex":
        """Create a new LlamaCloud managed index."""
        app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
        client = get_client(api_key, base_url, app_url, timeout)

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
            llama_parse_parameters=llama_parse_parameters or LlamaParseParameters(),
        )
        pipeline = client.pipelines.upsert_pipeline(
            project_id=project.id, request=pipeline_create
        )
        if pipeline.id is None:
            raise ValueError(f"Failed to create/get pipeline {name}")
        if verbose:
            print(f"Created pipeline {pipeline.id} with name {pipeline.name}")

        return cls(
            name,
            project_name=project.name,
            organization_id=project.organization_id,
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    async def acreate_index(
        cls: Type["LlamaCloudIndex"],
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        timeout: int = 60,
        verbose: bool = False,
        # ingestion configs
        embedding_config: Optional[PipelineCreateEmbeddingConfig] = None,
        transform_config: Optional[PipelineCreateTransformConfig] = None,
        llama_parse_parameters: Optional[LlamaParseParameters] = None,
        **kwargs: Any,
    ) -> "LlamaCloudIndex":
        """Create a new LlamaCloud managed index."""
        app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
        aclient = get_aclient(api_key, base_url, app_url, timeout)

        # create project if it doesn't exist
        project = await aclient.projects.upsert_project(
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
            llama_parse_parameters=llama_parse_parameters or LlamaParseParameters(),
        )
        pipeline = await aclient.pipelines.upsert_pipeline(
            project_id=project.id, request=pipeline_create
        )
        if pipeline.id is None:
            raise ValueError(f"Failed to create/get pipeline {name}")
        if verbose:
            print(f"Created pipeline {pipeline.id} with name {pipeline.name}")

        return cls(
            name,
            project_name=project.name,
            organization_id=project.organization_id,
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            **kwargs,
        )

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
        index = cls.create_index(
            name=name,
            project_name=project_name,
            organization_id=organization_id,
            api_key=api_key,
            base_url=base_url,
            app_url=app_url,
            timeout=timeout,
            verbose=verbose,
            embedding_config=embedding_config,
            transform_config=transform_config,
        )

        app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
        client = get_client(api_key, base_url, app_url, timeout)

        # this kicks off document ingestion
        upserted_documents = client.pipelines.upsert_batch_pipeline_documents(
            pipeline_id=index.pipeline.id,
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
        index.wait_for_completion(
            doc_ids=doc_ids, verbose=verbose, raise_on_error=raise_on_error
        )

        print(
            f"Find your index at {app_url}/project/{index.project.id}/deploy/{index.pipeline.id}"
        )

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
            self.wait_for_completion(
                doc_ids=[upserted_document.id], verbose=verbose, raise_on_error=True
            )

    async def ainsert(
        self, document: Document, verbose: bool = False, **insert_kwargs: Any
    ) -> None:
        """Insert a document."""
        with self._callback_manager.as_trace("insert"):
            upserted_documents = await self._aclient.pipelines.create_batch_pipeline_documents(
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
            await self.await_for_completion(
                doc_ids=[upserted_document.id], verbose=verbose, raise_on_error=True
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
            self.wait_for_completion(
                doc_ids=[upserted_document.id], verbose=verbose, raise_on_error=True
            )

    async def aupdate_ref_doc(
        self, document: Document, verbose: bool = False, **update_kwargs: Any
    ) -> None:
        """Upserts a document and its corresponding nodes."""
        with self._callback_manager.as_trace("update"):
            upserted_documents = await self._aclient.pipelines.upsert_batch_pipeline_documents(
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
            await self.await_for_completion(
                doc_ids=[upserted_document.id], verbose=verbose, raise_on_error=True
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
            self.wait_for_completion(doc_ids=doc_ids, verbose=True, raise_on_error=True)
            return [True] * len(doc_ids)

    async def arefresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed."""
        with self._callback_manager.as_trace("refresh"):
            upserted_documents = await self._aclient.pipelines.upsert_batch_pipeline_documents(
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
            await self.await_for_completion(
                doc_ids=doc_ids, verbose=True, raise_on_error=True
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

    async def adelete_ref_doc(
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
            await self._aclient.pipelines.delete_pipeline_document(
                pipeline_id=self.pipeline.id,
                document_id=quote_plus(quote_plus(ref_doc_id)),
            )
        except ApiError as e:
            if e.status_code == 404 and not raise_if_not_found:
                logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            else:
                raise

        # we have to wait for the pipeline instead of the document, because the document is already deleted
        await self.await_for_completion(verbose=verbose, raise_on_partial_success=False)

    def upload_file(
        self,
        file_path: str,
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

        # Add file to pipeline
        pipeline_file_create = PipelineFileCreate(file_id=file.id)
        self._client.pipelines.add_files_to_pipeline_api(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            self.wait_for_completion(
                file_ids=[file.id], verbose=verbose, raise_on_error=raise_on_error
            )
        return file.id

    async def aupload_file(
        self,
        file_path: str,
        verbose: bool = False,
        wait_for_ingestion: bool = True,
        raise_on_error: bool = False,
    ) -> str:
        """Upload a file to the index."""
        with open(file_path, "rb") as f:
            file = await self._aclient.files.upload_file(
                project_id=self.project.id, upload_file=f
            )
            if verbose:
                print(f"Uploaded file {file.id} with name {file.name}")

        # Add file to pipeline
        pipeline_file_create = PipelineFileCreate(file_id=file.id)
        await self._aclient.pipelines.add_files_to_pipeline_api(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            await self.await_for_completion(
                file_ids=[file.id], verbose=verbose, raise_on_error=raise_on_error
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
        self._client.pipelines.add_files_to_pipeline_api(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            self.wait_for_completion(
                file_ids=[file.id], verbose=verbose, raise_on_error=raise_on_error
            )
        return file.id

    async def aupload_file_from_url(
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
        file = await self._aclient.files.upload_file_from_url(
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
        await self._aclient.pipelines.add_files_to_pipeline_api(
            pipeline_id=self.pipeline.id, request=[pipeline_file_create]
        )

        if wait_for_ingestion:
            await self.await_for_completion(
                file_ids=[file.id], verbose=verbose, raise_on_error=raise_on_error
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
