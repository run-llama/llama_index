"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import logging
from typing import Any, Dict, Optional, Sequence, Type

import requests

from llama_index.core.base_retriever import BaseRetriever
from llama_index.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.schema import BaseNode, Document
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext

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
        cluster_id (str): Zilliz Cloud's cluster ID.
        token (str): Zilliz Cloud's token.
        cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
        pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None
        collection_name (str='zcp_llamalection'): A collection name, defaults to 'zcp_llamalection'. If no pipeline_ids is given, get or create pipelines with collection_name.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    def __init__(
        self,
        cluster_id: str,
        token: str,
        cloud_region: str = "gcp-us-west1",
        pipeline_ids: Optional[Dict] = None,
        collection_name: str = "zcp_llamalection",
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        if pipeline_ids is None:
            pipeline_ids = self._get_pipeline_ids(
                cluster_id=cluster_id,
                token=token,
                cloud_region=cloud_region,
                collection_name=collection_name,
            )
        if len(pipeline_ids) == 0:
            pipeline_ids = self._create_pipelines(
                cluster_id=cluster_id,
                token=token,
                cloud_region=cloud_region,
                collection_name=collection_name,
            )
        assert set(PIPELINE_TYPES).issubset(
            set(pipeline_ids.keys())
        ), f"Missing pipeline(s): {set(PIPELINE_TYPES) - set(pipeline_ids.keys())}"

        index_struct = ZillizCloudPipelineIndexStruct(
            index_id="_".join([str(x) for x in pipeline_ids.values()]),
            summary="Zilliz Cloud Pipeline Index",
        )

        super().__init__(
            show_progress=show_progress, index_struct=index_struct, **kwargs
        )

        domain = f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"

        ingest_pipe_id = pipeline_ids["INGESTION"]
        search_pipe_id = pipeline_ids["SEARCH"]
        deletion_pipe_id = pipeline_ids["DELETION"]
        self.ingestion_url = f"{domain}/{ingest_pipe_id}/run"
        self.search_url = f"{domain}/{search_pipe_id}/run"
        self.deletion_url = f"{domain}/{deletion_pipe_id}/run"

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        self.cluster_id = cluster_id

    def insert_doc_url(self, url: str, metadata: Optional[Dict] = None) -> None:
        if metadata is None:
            metadata = {}
        params = {"data": {"doc_url": url}}
        params["data"].update(metadata)
        response = requests.post(self.ingestion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever."""
        from llama_index.indices.managed.zilliz.retriever import (
            ZillizCloudPipelineRetriever,
        )

        return ZillizCloudPipelineRetriever(self, **kwargs)

    @classmethod
    def from_document_url(
        cls,
        url: str,
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
            url: a gcs or s3 signed url
            cluster_id (str): Zilliz Cloud's cluster ID.
            token (str): Zilliz Cloud's token.
            cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
            pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None
            collection_name (str='zcp_llamalection'): A collection name, defaults to 'zcp_llamalection'. If no pipeline_ids is given, get or create pipelines with collection_name.
            metadata (Dict=None): A dictionary of metadata. Defaults to None. The key must be string and the value must be a string, float, integer, or boolean.
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            An initialized ZillizCloudPipelineIndex
        """
        if metadata is None:
            metadata = {}
        if pipeline_ids is None:
            pipeline_ids = cls._get_pipeline_ids(
                cluster_id=cluster_id,
                token=token,
                cloud_region=cloud_region,
                collection_name=collection_name,
            )
        if len(pipeline_ids) == 0:
            pipeline_ids = cls._create_pipelines(
                cluster_id=cluster_id,
                token=token,
                cloud_region=cloud_region,
                collection_name=collection_name,
                metadata_schema={k: get_zcp_type(v) for k, v in metadata.items()},
            )
        index = cls(
            cluster_id=cluster_id,
            token=token,
            cloud_region=cloud_region,
            pipeline_ids=pipeline_ids,
            collection_name=collection_name,
            show_progress=show_progress,
            **kwargs,
        )
        try:
            index.insert_doc_url(url=url, metadata=metadata)
        except Exception as e:
            logger.error(
                "Failed to build managed index given document url (%s):\n%s", url, e
            )
        return index

    @classmethod
    def _get_pipeline_ids(
        cls, cluster_id: str, token: str, cloud_region: str, collection_name: str
    ) -> dict:
        """Get pipeline ids."""
        url = f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Get pipelines
        response = requests.get(url, headers=headers)
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
                if cluster_id in pipe_clusters and collection_name in pipe_collections:
                    pipeline_ids[pipe_type] = pipe_id
            elif pipe_type == "INGESTION":
                if (
                    cluster_id == pipe_info["clusterId"]
                    and collection_name == pipe_info["newCollectionName"]
                ):
                    pipeline_ids[pipe_type] = pipe_id
            elif pipe_type == "DELETION":
                if (
                    cluster_id == pipe_info["clusterId"]
                    and collection_name == pipe_info["collectionName"]
                ):
                    pipeline_ids[pipe_type] = pipe_id
        return pipeline_ids

    @classmethod
    def _create_pipelines(
        cls,
        cluster_id: str,
        token: str,
        cloud_region: str,
        collection_name: str,
        metadata_schema: Optional[Dict] = None,
    ) -> dict:
        """Given collection_name, create INGESTION, SEARCH, DELETION pipelines."""
        url = f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        params_dict = {}
        functions = [
            {
                "name": "index_my_doc",
                "action": "INDEX_DOC",
                "inputField": "doc_url",
                "language": "ENGLISH",
            }
        ]
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
            "name": "llamaindex_ingestion",
            "clusterId": cluster_id,
            "newCollectionName": collection_name,
            "type": "INGESTION",
            "functions": functions,
        }

        params_dict["SEARCH"] = {
            "name": "llamaindex_search",
            "type": "SEARCH",
            "functions": [
                {
                    "name": "search_chunk_text",
                    "action": "SEARCH_DOC_CHUNK",
                    "inputField": "query_text",
                    "clusterId": cluster_id,
                    "collectionName": collection_name,
                }
            ],
        }

        params_dict["DELETION"] = {
            "name": "llamaindex_deletion",
            "type": "DELETION",
            "functions": [
                {
                    "name": "purge_chunks_by_doc_name",
                    "action": "PURGE_DOC_INDEX",
                    "inputField": "doc_name",
                }
            ],
            "clusterId": cluster_id,
            "collectionName": collection_name,
        }

        pipeline_ids = {}
        for k, v in params_dict.items():
            response = requests.post(url, headers=headers, json=v)
            if response.status_code != 200:
                raise RuntimeError(response.text)
            response_dict = response.json()
            if response_dict["code"] != 200:
                raise RuntimeError(response_dict)
            pipeline_ids[k] = response_dict["data"]["pipelineId"]

        return pipeline_ids

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
