"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import logging
from typing import Any, Dict, Optional, Sequence, Type

import requests
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.core.schema import BaseNode, Document

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
        pipeline_ids (dict): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION.
        api_key (str): Zilliz Cloud's API key.
        cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    def __init__(
        self,
        pipeline_ids: Dict,
        api_key: str = None,
        cloud_region: str = "gcp-us-west1",
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        self.token = api_key
        self.cloud_region = cloud_region
        self.domain = (
            f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"
        )
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.pipeline_ids = pipeline_ids or {}

        if len(self.pipeline_ids) == 0:
            print(
                "Pipeline ids are required. You can use the classmethod `ZillizCloudPipelineIndex.create_pipelines` to create pipelines and get pipeline ids."
            )
        else:
            assert set(PIPELINE_TYPES).issubset(
                set(self.pipeline_ids.keys())
            ), f"Missing pipeline(s): {set(PIPELINE_TYPES) - set(self.pipeline_ids.keys())}"

        index_struct = ZillizCloudPipelineIndexStruct(
            index_id="-".join(pipeline_ids.values()),
            summary="Zilliz Cloud Pipeline Index",
        )

        super().__init__(
            show_progress=show_progress, index_struct=index_struct, **kwargs
        )

    def _insert_doc_url(self, url: str, metadata: Optional[Dict] = None) -> None:
        """Insert doc from url with an initialized index using doc pipelines."""
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

    def _insert(self, nodes: Sequence[BaseNode], metadata: Optional[Dict] = None):
        """Insert doc from text nodes with an initialized index using text pipelines."""
        ingest_pipe_id = self.pipeline_ids.get("INGESTION")
        ingestion_url = f"{self.domain}/{ingest_pipe_id}/run"

        text_list = [n.get_content() for n in nodes]
        if metadata is None:
            metadata = {}
        params = {"data": {"text_list": text_list}}
        params["data"].update(metadata)
        response = requests.post(ingestion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        return response_dict["data"]

    def delete_by_expression(self, expression: str):
        """Delete data by Milvus boolean expression if using the corresponding deletion pipeline."""
        deletion_pipe_id = self.pipeline_ids.get("DELETION")
        deletion_url = f"{self.domain}/{deletion_pipe_id}/run"

        params = {"data": {"expression": expression}}
        response = requests.post(deletion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        return response_dict["data"]

    def delete_by_doc_name(self, doc_name: str):
        """Delete data by doc name if using the corresponding deletion pipeline."""
        deletion_pipe_id = self.pipeline_ids.get("DELETION")
        deletion_url = f"{self.domain}/{deletion_pipe_id}/run"

        params = {"data": {"doc_name": doc_name}}
        response = requests.post(deletion_url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        return response_dict["data"]

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

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever."""
        from llama_index.indices.managed.zilliz.retriever import (
            ZillizCloudPipelineRetriever,
        )

        return ZillizCloudPipelineRetriever(self, **kwargs)

    @staticmethod
    def create_pipelines(
        project_id: str,
        cluster_id: str,
        cloud_region: str = "gcp-us-west1",
        api_key: str = None,
        collection_name: str = "zcp_llamalection",
        data_type: str = "text",
        metadata_schema: Optional[Dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Create INGESTION, SEARCH, DELETION pipelines using self.collection_name.

        Args:
            project_id (str): Zilliz Cloud's project ID.
            cluster_id (str): Zilliz Cloud's cluster ID.
            api_key (str=None): Zilliz Cloud's API key. Defaults to None.
            cloud_region (str='gcp-us-west1'): The region of Zilliz Cloud's cluster. Defaults to 'gcp-us-west1'.
            collection_name (str="zcp_llamalection"): A collection name, defaults to 'zcp_llamalection'.
            data_type (str="text"): The data type of pipelines, defaults to "text". Currently only "text" or "doc" are supported.
            metadata_schema (Dict=None): A dictionary of metadata schema, defaults to None. Use metadata name as key and the corresponding data type as value: {'field_name': 'field_type'}.
                Only support the following values as the field type: 'Bool', 'Int8', 'Int16', 'Int32', 'Int64', 'Float', 'Double', 'VarChar'.
            kwargs: optional function parameters to create ingestion & search pipelines.
                - language: The language of documents. Available options: "ENGLISH", "CHINESE".
                - embedding: The embedding service used in both ingestion & search pipeline.
                - reranker: The reranker service used in search function.
                - chunkSize: The chunk size to split a document. Only for doc data.
                - splitBy: The separators to chunking a document. Only for doc data.

        Returns:
            The pipeline ids of created pipelines.

        Example:
            >>> from llama_index.indices import ZillizCloudPipelineIndex
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
        if data_type == "text":
            ingest_action = "INDEX_TEXT"
            search_action = "SEARCH_TEXT"
        elif data_type == "doc":
            ingest_action = "INDEX_DOC"
            search_action = "SEARCH_DOC_CHUNK"
        else:
            raise Exception("Only text or doc is supported as the data type.")

        params_dict = {}
        additional_params = kwargs or {}

        language = additional_params.pop("language", "ENGLISH")
        embedding = additional_params.pop("embedding", "zilliz/bge-base-en-v1.5")
        reranker = additional_params.pop("reranker", None)
        index_func = {
            "name": "llamaindex_index",
            "action": ingest_action,
            "language": language,
            "embedding": embedding,
        }
        index_func.update(additional_params)
        ingest_functions = [index_func]
        if metadata_schema:
            for k, v in metadata_schema.items():
                preserve_func = {
                    "name": f"keep_{k}",
                    "action": "PRESERVE",
                    "inputField": k,
                    "outputField": k,
                    "fieldType": v,
                }
                ingest_functions.append(preserve_func)
        params_dict["INGESTION"] = {
            "name": f"{collection_name}_ingestion",
            "projectId": project_id,
            "clusterId": cluster_id,
            "collectionName": collection_name,
            "type": "INGESTION",
            "functions": ingest_functions,
        }

        search_function = {
            "name": "llamaindex_search",
            "action": search_action,
            "clusterId": cluster_id,
            "collectionName": collection_name,
            "embedding": embedding,
        }
        if reranker:
            search_function["reranker"] = reranker
        params_dict["SEARCH"] = {
            "name": f"{collection_name}_search",
            "projectId": project_id,
            "type": "SEARCH",
            "functions": [search_function],
        }

        params_dict["DELETION"] = {
            "name": f"{collection_name}_deletion",
            "type": "DELETION",
            "functions": [
                {
                    "name": "purge_by_expression",
                    "action": "PURGE_BY_EXPRESSION",
                }
            ],
            "projectId": project_id,
            "clusterId": cluster_id,
            "collectionName": collection_name,
        }

        domain = f"https://controller.api.{cloud_region}.zillizcloud.com/v1/pipelines"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        pipeline_ids = {}

        for k, v in params_dict.items():
            response = requests.post(domain, headers=headers, json=v)
            if response.status_code != 200:
                raise RuntimeError(response.text)
            response_dict = response.json()
            if response_dict["code"] != 200:
                raise RuntimeError(response_dict)
            pipeline_ids[k] = response_dict["data"]["pipelineId"]

        return pipeline_ids

    @classmethod
    def from_document_url(
        cls,
        url: str,
        pipeline_ids: Optional[Dict] = None,
        api_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> BaseManagedIndex:
        """Zilliz Cloud Pipeline loads document from a signed url and then builds auto index for it.

        Args:
            url: a gcs or s3 signed url.
            pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None.
            api_key (str): Zilliz Cloud's API Key.
            metadata (Dict=None): A dictionary of metadata. Defaults to None. The key must be string and the value must be a string, float, integer, or boolean.
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            An initialized ZillizCloudPipelineIndex

        Example:
            >>> from llama_index.indices import ZillizCloudPipelineIndex
            >>> api_key = "{YOUR_ZILLIZ_CLOUD_API_KEY}"
            >>> pipeline_ids = ZillizCloudPipelineIndex.create_pipelines(
            >>>     project_id="{YOUR_ZILLIZ_PROJECT_ID}",
            >>>     cluster_id="{YOUR_ZILLIZ_CLUSTER_ID}",
            >>>     api_key=api_key,
            >>>     data_type="doc"
            >>> )
            >>> ZillizCloudPipelineIndex.from_document_url(
            >>>     url='https://oss_bucket.test_doc.ext',
            >>>     pipeline_ids=pipeline_ids,
            >>>     api_key=api_key
            >>> )
        """
        metadata = metadata or {}
        index = cls(
            pipeline_ids=pipeline_ids,
            api_key=api_key,
            show_progress=show_progress,
            **kwargs,
        )

        try:
            index._insert_doc_url(url=url, metadata=metadata)
        except Exception as e:
            logger.error(
                "Failed to build managed index given document url (%s):\n%s", url, e
            )
        return index

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        pipeline_ids: Optional[Dict] = None,
        api_key: Optional[str] = None,
        show_progress: bool = False,
        metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Zilliz Cloud Pipeline index from a sequence of documents.

        Args:
            documents: a sequence of llamaindex documents.
            pipeline_ids (dict=None): A dictionary of pipeline ids for INGESTION, SEARCH, DELETION. Defaults to None.
            api_key (str): Zilliz Cloud's API Key.
            metadata (Dict=None): A dictionary of metadata. Defaults to None. The key must be string and the value must be a string, float, integer, or boolean.
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

        Returns:
            An initialized ZillizCloudPipelineIndex

        Example:
            >>> from llama_index.indices import ZillizCloudPipelineIndex
            >>> api_key = "{YOUR_ZILLIZ_CLOUD_API_KEY}"
            >>> pipeline_ids = ZillizCloudPipelineIndex.create_pipelines(
            >>>     project_id="{YOUR_ZILLIZ_PROJECT_ID}",
            >>>     cluster_id="{YOUR_ZILLIZ_CLUSTER_ID}",
            >>>     api_key=api_key,
            >>>     data_type="text"
            >>> )
            >>> ZillizCloudPipelineIndex.from_documents(
            >>>     documents=my_documents,
            >>>     pipeline_ids=pipeline_ids,
            >>>     api_key=api_key
            >>> )
        """
        metadata = metadata or {}
        index = cls(
            pipeline_ids=pipeline_ids,
            api_key=api_key,
            show_progress=show_progress,
            **kwargs,
        )

        try:
            index._insert(nodes=documents, metadata=metadata)
        except Exception as e:
            logger.error("Failed to build managed index given documents:\n%s", e)
        return index

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        raise NotImplementedError(
            "Building index from nodes is not yet supported with Zilliz Cloud Pipeline."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "Deleting nodes is not yet supported with Zilliz Cloud Pipeline."
        )
