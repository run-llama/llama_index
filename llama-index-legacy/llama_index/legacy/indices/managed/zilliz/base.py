"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import logging
from typing import Any, Dict, Optional, Sequence, Type

import requests

from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.legacy.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.legacy.schema import BaseNode, Document
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)

PIPELINE_TYPES = ["INGESTION", "SEARCH", "DELETION"]


def get_zcp_type(value: Any) -> str:
    if isinstance(value, str):
        return "VarChar"
    elif isinstance(value, bool):
        return "Bool"
    elif isinstance(value, int):
        return "Int64"
    elif isinstance(value, float):
        return "Double"
    else:
        raise TypeError(
            "Invalid data type of metadata: must be str, bool, int, or float."
        )


class ZillizCloudPipelineIndexStruct(IndexDict):
    """Zilliz Cloud Pipeline's Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.ZILLIZ_CLOUD_PIPELINE


class ZillizCloudPipelineIndex(BaseManagedIndex):
    """Zilliz Cloud Pipeline's Index.

    The Zilliz Cloud Pipeline's index implements a managed index that uses Zilliz Cloud Pipelines as the backend.

    Args:
        project_id (str): Zilliz Cloud's project ID.
        cluster_id (str): Zilliz Cloud's cluster ID.
        token (str): Zilliz Cloud's token.
        cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
        pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None.
        collection_name (str='zcp_llamalection'): A collection name, defaults to 'zcp_llamalection'. If no pipeline_ids is given, get pipelines with collection_name.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    def __init__(
        self,
        project_id: str,
        cluster_id: str,
        token: str,
        cloud_region: str = "gcp-us-west1",
        pipeline_ids: Optional[Dict] = None,
        collection_name: str = "zcp_llamalection",
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        self.project_id = project_id
        self.cluster_id = cluster_id
        self.token = token
        self.cloud_region = cloud_region
        self.collection_name = collection_name
        self.domain = (
            f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"
        )
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.pipeline_ids = pipeline_ids or self.get_pipeline_ids()

        index_struct = ZillizCloudPipelineIndexStruct(
            index_id=collection_name,
            summary="Zilliz Cloud Pipeline Index",
        )

        super().__init__(
            show_progress=show_progress, index_struct=index_struct, **kwargs
        )

        if len(self.pipeline_ids) == 0:
            print("No available pipelines. Please create pipelines first.")
        else:
            assert set(PIPELINE_TYPES).issubset(
                set(self.pipeline_ids.keys())
            ), f"Missing pipeline(s): {set(PIPELINE_TYPES) - set(self.pipeline_ids.keys())}"

    def insert_doc_url(self, url: str, metadata: Optional[Dict] = None) -> None:
        """Insert doc from url with an initialized index.


        Example:
        >>> from llama_index.legacy.indices import ZillizCloudPipelineIndex
        >>> index = ZillizCloudPipelineIndex(
        >>>     project_id='YOUR_ZILLIZ_CLOUD_PROJECT_ID',
        >>>     cluster_id='YOUR_ZILLIZ_CLOUD_CLUSTER_ID',
        >>>     token='YOUR_ZILLIZ_CLOUD_API_KEY',
        >>>     collection_name='your_collection_name'
        >>> )
        >>> index.insert_doc_url(
        >>>     url='https://oss_bucket.test_doc.ext',
        >>>     metadata={'year': 2023, 'author': 'zilliz'}  # only required when the Index was created with metadata schemas
        >>> )
        """
        ingest_pipe_id = self.pipeline_ids.get("INGESTION")
        ingestion_url = f"{self.domain}/{ingest_pipe_id}/run"

        if metadata is None:
            metadata = {}
        params = {"data": {"doc_url": url}}
        params["data"].update(metadata)
        response = requests.post(ingestion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        return response_dict["data"]

    def delete_by_doc_name(self, doc_name: str) -> int:
        deletion_pipe_id = self.pipeline_ids.get("DELETION")
        deletion_url = f"{self.domain}/{deletion_pipe_id}/run"

        params = {"data": {"doc_name": doc_name}}
        response = requests.post(deletion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        try:
            return response_dict["data"]
        except Exception as e:
            raise RuntimeError(f"Run Zilliz Cloud Pipelines failed: {e}")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever."""
        from llama_index.legacy.indices.managed.zilliz.retriever import (
            ZillizCloudPipelineRetriever,
        )

        return ZillizCloudPipelineRetriever(self, **kwargs)

    def get_pipeline_ids(self) -> dict:
        """Get pipeline ids."""
        url = f"{self.domain}?projectId={self.project_id}"

        # Get pipelines
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        data = response_dict["data"]
        pipeline_ids = {}
        for pipe_info in data:
            pipe_id = pipe_info["pipelineId"]
            pipe_type = pipe_info["type"]

            if pipe_type == "SEARCH":
                pipe_clusters = [x["clusterId"] for x in pipe_info["functions"]]
                pipe_collections = [x["collectionName"] for x in pipe_info["functions"]]
                if (
                    self.cluster_id in pipe_clusters
                    and self.collection_name in pipe_collections
                ):
                    pipeline_ids[pipe_type] = pipe_id
            elif pipe_type == "INGESTION":
                if (
                    self.cluster_id == pipe_info["clusterId"]
                    and self.collection_name == pipe_info["newCollectionName"]
                ):
                    pipeline_ids[pipe_type] = pipe_id
            elif pipe_type == "DELETION":
                if (
                    self.cluster_id == pipe_info["clusterId"]
                    and self.collection_name == pipe_info["collectionName"]
                ):
                    pipeline_ids[pipe_type] = pipe_id
        return pipeline_ids

    def create_pipelines(
        self, metadata_schema: Optional[Dict] = None, **kwargs: str
    ) -> dict:
        """Create INGESTION, SEARCH, DELETION pipelines using self.collection_name.

        Args:
            metadata_schema (Dict=None): A dictionary of metadata schema, defaults to None. Use metadata name as key and the corresponding data type as value: {'field_name': 'field_type'}.
                Only support the following values as the field type: 'Bool', 'Int8', 'Int16', 'Int32', 'Int64', 'Float', 'Double', 'VarChar'.
            kwargs: optional parameters to create ingestion pipeline
                - chunkSize: An integer within range [20, 500] to customize chunk size.
                - language: The language of documents. Available options: "ENGLISH", "CHINESE".

        Returns:
            A dictionary of pipeline ids for INGESTION, SEARCH, and DELETION pipelines.

        Example:
            >>> from llama_index.legacy.indices import ZillizCloudPipelineIndex
            >>> index = ZillizCloudPipelineIndex(
            >>>     project_id='YOUR_ZILLIZ_CLOUD_PROJECT_ID',
            >>>     cluster_id='YOUR_ZILLIZ_CLOUD_CLUSTER_ID',
            >>>     token='YOUR_ZILLIZ_CLOUD_API_KEY',
            >>>     collection_name='your_new_collection_name'
            >>> )
            >>> pipeline_ids = index.create_pipelines(
            >>>     metadata_schema={'year': 'Int32', 'author': 'VarChar'}  # optional, defaults to None
            >>> )
        """
        if len(self.pipeline_ids) > 0:
            raise RuntimeError(
                f"Pipelines already exist for collection {self.collection_name}: {self.pipeline_ids}"
            )

        params_dict = {}
        index_doc_func = {
            "name": "index_my_doc",
            "action": "INDEX_DOC",
            "inputField": "doc_url",
            "language": "ENGLISH",
        }
        index_doc_func.update(kwargs)
        functions = [index_doc_func]
        if metadata_schema:
            for k, v in metadata_schema.items():
                preserve_func = {
                    "name": f"keep_{k}",
                    "action": "PRESERVE",
                    "inputField": k,
                    "outputField": k,
                    "fieldType": v,
                }
                functions.append(preserve_func)
        params_dict["INGESTION"] = {
            "name": f"{self.collection_name}_ingestion",
            "projectId": self.project_id,
            "clusterId": self.cluster_id,
            "newCollectionName": self.collection_name,
            "type": "INGESTION",
            "functions": functions,
        }

        params_dict["SEARCH"] = {
            "name": f"{self.collection_name}_search",
            "projectId": self.project_id,
            "type": "SEARCH",
            "functions": [
                {
                    "name": "search_chunk_text",
                    "action": "SEARCH_DOC_CHUNK",
                    "inputField": "query_text",
                    "clusterId": self.cluster_id,
                    "collectionName": self.collection_name,
                }
            ],
        }

        params_dict["DELETION"] = {
            "name": f"{self.collection_name}_deletion",
            "type": "DELETION",
            "functions": [
                {
                    "name": "purge_chunks_by_doc_name",
                    "action": "PURGE_DOC_INDEX",
                    "inputField": "doc_name",
                }
            ],
            "projectId": self.project_id,
            "clusterId": self.cluster_id,
            "collectionName": self.collection_name,
        }

        for k, v in params_dict.items():
            response = requests.post(self.domain, headers=self.headers, json=v)
            if response.status_code != 200:
                raise RuntimeError(response.text)
            response_dict = response.json()
            if response_dict["code"] != 200:
                raise RuntimeError(response_dict)
            self.pipeline_ids[k] = response_dict["data"]["pipelineId"]

        return self.pipeline_ids

    @classmethod
    def from_document_url(
        cls,
        url: str,
        project_id: str,
        cluster_id: str,
        token: str,
        cloud_region: str = "gcp-us-west1",
        pipeline_ids: Optional[Dict] = None,
        collection_name: str = "zcp_llamalection",
        metadata: Optional[Dict] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> BaseManagedIndex:
        """Zilliz Cloud Pipeline loads document from a signed url and then builds auto index for it.

        Args:
            url: a gcs or s3 signed url.
            project_id (str): Zilliz Cloud's project ID.
            cluster_id (str): Zilliz Cloud's cluster ID.
            token (str): Zilliz Cloud's token.
            cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
            pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None.
            collection_name (str='zcp_llamalection'): A collection name, defaults to 'zcp_llamalection'. If no pipeline_ids is given, get or create pipelines with collection_name.
            metadata (Dict=None): A dictionary of metadata. Defaults to None. The key must be string and the value must be a string, float, integer, or boolean.
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            An initialized ZillizCloudPipelineIndex

        Example:
            >>> from llama_index.legacy.indices import ZillizCloudPipelineIndex
            >>> index = ZillizCloudPipelineIndex.from_document_url(
            >>>     url='https://oss_bucket.test_doc.ext',
            >>>     project_id='YOUR_ZILLIZ_CLOUD_PROJECT_ID',
            >>>     cluster_id='YOUR_ZILLIZ_CLOUD_CLUSTER_ID',
            >>>     token='YOUR_ZILLIZ_CLOUD_API_KEY',
            >>>     collection_name='your_collection_name'
            >>> )
        """
        metadata = metadata or {}
        index = cls(
            project_id=project_id,
            cluster_id=cluster_id,
            token=token,
            cloud_region=cloud_region,
            pipeline_ids=pipeline_ids,
            collection_name=collection_name,
            show_progress=show_progress,
            **kwargs,
        )
        if len(index.pipeline_ids) == 0:
            index.pipeline_ids = index.create_pipelines(
                metadata_schema={k: get_zcp_type(v) for k, v in metadata.items()}
            )
            print("Pipelines are automatically created.")

        try:
            index.insert_doc_url(url=url, metadata=metadata)
        except Exception as e:
            logger.error(
                "Failed to build managed index given document url (%s):\n%s", url, e
            )
        return index

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError(
            "Inserting nodes is not yet supported with Zilliz Cloud Pipeline."
        )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        raise NotImplementedError(
            "Deleting a reference document is not yet supported with Zilliz Cloud Pipeline."
        )

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        raise NotImplementedError(
            "Updating referenced document is not yet supported with Zilliz Cloud Pipeline."
        )

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Zilliz Cloud Pipeline index from a sequence of documents."""
        raise NotImplementedError(
            "Loading from document texts is not yet supported with Zilliz Cloud Pipeline."
        )

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        raise NotImplementedError(
            "Building index from nodes is not yet supported with Zilliz Cloud Pipeline."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "Deleting nodes is not yet supported with Zilliz Cloud Pipeline."
        )
