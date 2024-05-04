"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""
import os
import time
from typing import Any, List, Optional, Sequence, Type
from enum import Enum
import time
import requests
import json

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.settings import Settings

from llama_index.indices.managed.dashscope.api_utils import (
    get_pipeline_create,
    default_transformations,
)

from llama_index.indices.managed.dashscope.constants import (
    DASHSCOPE_DEFAULT_BASE_URL,
    UPSERT_PIPELINE_ENDPOINT,
    START_PIPELINE_ENDPOINT,
    CHECK_INGESTION_ENDPOINT,
)


class Status(Enum):
    ERROR = "ERROR"
    SUCCESS = "Success"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELED = "CANCELED"
    FAILED = "FAILED"
    FINISHED = "FINISHED"


class DashScopeCloudIndex(BaseManagedIndex):
    """DashScope Cloud Platform Index."""

    def __init__(
        self,
        name: str,
        nodes: Optional[List[BaseNode]] = None,
        transformations: Optional[List[TransformComponent]] = None,
        timeout: int = 60,
        workspace_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = DASHSCOPE_DEFAULT_BASE_URL,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Platform Index."""
        self.name = name
        self.transformations = transformations or []

        if nodes is not None:
            raise ValueError(
                "DashScopeCloudIndex does not support nodes on initialization"
            )

        self.workspace_id = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID")
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self._base_url = os.environ.get("DASHSCOPE_BASE_URL", None) or base_url
        self._timeout = timeout
        self._show_progress = show_progress
        self._service_context = None
        self._callback_manager = callback_manager or Settings.callback_manager

    @classmethod
    def from_documents(  # type: ignore
        cls: Type["DashScopeCloudIndex"],
        documents: List[Document],
        name: str,
        transformations: Optional[List[TransformComponent]] = None,
        workspace_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DashScopeCloudIndex":
        """Build a DashScope index from a sequence of documents."""
        pipeline_create = get_pipeline_create(
            name, transformations or default_transformations(), documents
        )
        # for debug
        json.dump(
            pipeline_create, open("pipeline_create.json", "w"), ensure_ascii=False
        )

        workspace_id = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID")
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        base_url = (
            base_url
            or os.environ.get("DASHSCOPE_BASE_URL", None)
            or DASHSCOPE_DEFAULT_BASE_URL
        )
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "utf-8",
            "X-DashScope-WorkSpace": workspace_id,
            "Authorization": "Bearer " + api_key,
            "X-DashScope-OpenAPISource": "CloudSDK",
            # for debug
            # 'X-DashScope-ApiKeyId': 'test_api_key_id_123456',
            # 'X-DashScope-Uid': "test_uid_123456",
            # "X-DashScope-SubUid": "test_sub_uid_123456"
        }
        print(base_url + UPSERT_PIPELINE_ENDPOINT)
        print(json.dumps(pipeline_create))
        response = requests.put(
            base_url + UPSERT_PIPELINE_ENDPOINT,
            data=json.dumps(pipeline_create),
            headers=headers,
        )
        response_text = response.json()
        pipeline_id = response_text.get("id", None)

        if response_text.get("code", "") != Status.SUCCESS.value or pipeline_id is None:
            raise ValueError(
                f"Failed to create index: {response_text.get('message', '')}\n{response_text}"
            )
        if verbose:
            print(f"Starting creating index {name}, pipeline_id: {pipeline_id}")

        print(base_url + START_PIPELINE_ENDPOINT.format(pipeline_id=pipeline_id))

        response = requests.post(
            base_url + START_PIPELINE_ENDPOINT.format(pipeline_id=pipeline_id),
            headers=headers,
        )
        response_text = response.json()
        ingestion_id = response_text.get("ingestionId", None)

        if (
            response_text.get("code", "") != Status.SUCCESS.value
            or ingestion_id is None
        ):
            raise ValueError(
                f"Failed to start ingestion: {response_text.get('message', '')}\n{response_text}"
            )
        if verbose:
            print(f"Starting ingestion for index {name}, ingestion_id: {ingestion_id}")

        ingestion_status = ""
        failed_docs = []

        while True:
            print(
                base_url
                + CHECK_INGESTION_ENDPOINT.format(
                    pipeline_id=pipeline_id, ingestion_id=ingestion_id
                )
            )
            response = requests.get(
                base_url
                + CHECK_INGESTION_ENDPOINT.format(
                    pipeline_id=pipeline_id, ingestion_id=ingestion_id
                ),
                headers=headers,
            )
            try:
                response_text = response.json()
            except Exception as e:
                print(f"Failed to get response: \n{response.text}\nretrying...")
                continue

            if response_text.get("code", "") != Status.SUCCESS.value:
                print(
                    f"Failed to get ingestion status: {response_text.get('message', '')}\n{response_text}\nretrying..."
                )
                continue
            ingestion_status = response_text.get("ingestion_status", "")
            failed_docs = response_text.get("failed_docs", "")
            if verbose:
                print(f"Current status: {ingestion_status}")
            if ingestion_status in ["COMPLETED", "FAILED"]:
                break
            time.sleep(5)

        if verbose:
            print(f"ingestion_status {ingestion_status}")
            print(f"failed_docs: {failed_docs}")

        if ingestion_status == "FAILED":
            print("Index {name} created failed!")
            return None

        if verbose:
            print(f"Index {name} created successfully!")

        return cls(
            name,
            transformations=transformations,
            workspace_id=workspace_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.dashscope.retriever import (
            DashScopeCloudRetriever,
        )

        return DashScopeCloudRetriever(
            self.name,
            **kwargs,
        )

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        kwargs["retriever"] = self.as_retriever(**kwargs)
        return RetrieverQueryEngine.from_args(**kwargs)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of documents (each a node)."""
        raise NotImplementedError("_insert not implemented.")

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        raise NotImplementedError("delete_ref_doc not implemented.")

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes."""
        raise NotImplementedError("update_ref_doc not implemented.")
