"""
Vertex AI Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, List, Optional, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY, node_to_metadata_dict
from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
    VectorSearchSDKManager,
)
from llama_index.vector_stores.vertexaivectorsearch import utils
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud import storage

_logger = logging.getLogger(__name__)


class VertexAIVectorStore(BasePydanticVectorStore):
    """
    Vertex AI Vector Search vector store.

    In this vector store, embeddings are stored in Vertex AI Vector Store and
    docs are stored within Cloud Storage bucket.

    During query time, the index uses Vertex AI Vector Search to query for the
    top k most similar nodes.

    Args:
        project_id (str) : The Google Cloud Project ID.
        region (str)     : The default location making the API calls.
                           It must be the same location as where Vector Search
                           index created and must be regional.
        index_id (str)   : The fully qualified resource name of the created
                           index in Vertex AI Vector Search.
        endpoint_id (str): The fully qualified resource name of the created
                           index endpoint in Vertex AI Vector Search.
        gcs_bucket_name (Optional[str]):
                           The location where the vectors will be stored for
                           the index to be created in batch mode.
        credentials_path (Optional[str]):
                           The path of the Google credentials on the local file
                           system.

    Examples:
        `pip install llama-index-vector-stores-vertexaivectorsearch`

        ```python
        from
        vector_store = VertexAIVectorStore(
            project_id=PROJECT_ID,
            region=REGION,
            index_id="<index_resource_name>"
            endpoint_id="<index_endpoint_resource_name>"
        )
        ```

    """

    stores_text: bool = True
    remove_text_from_metadata: bool = True
    flat_metadata: bool = False

    text_key: str

    project_id: str
    region: str
    index_id: str
    endpoint_id: str
    gcs_bucket_name: Optional[str] = None
    credentials_path: Optional[str] = None

    _index: MatchingEngineIndex = PrivateAttr()
    _endpoint: MatchingEngineIndexEndpoint = PrivateAttr()
    _index_metadata: dict = PrivateAttr()
    _stream_update: bool = PrivateAttr()
    _staging_bucket: storage.Bucket = PrivateAttr()
    # _document_storage: GCSDocumentStorage = PrivateAttr()

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        gcs_bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        remove_text_from_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            index_id=index_id,
            endpoint_id=endpoint_id,
            gcs_bucket_name=gcs_bucket_name,
            credentials_path=credentials_path,
            text_key=text_key,
            remove_text_from_metadata=remove_text_from_metadata,
        )

        """Initialize params."""
        _sdk_manager = VectorSearchSDKManager(
            project_id=project_id, region=region, credentials_path=credentials_path
        )

        # get index and endpoint resource names including metadata
        self._index = _sdk_manager.get_index(index_id=index_id)
        self._endpoint = _sdk_manager.get_endpoint(endpoint_id=endpoint_id)
        self._index_metadata = self._index.to_dict()

        # get index update method from index metadata
        self._stream_update = False
        if self._index_metadata["indexUpdateMethod"] == "STREAM_UPDATE":
            self._stream_update = True

        # get bucket object when available
        if self.gcs_bucket_name:
            self._staging_bucket = _sdk_manager.get_gcs_bucket(
                bucket_name=gcs_bucket_name
            )
        else:
            self._staging_bucket = None

    @classmethod
    def from_params(
        cls,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        gcs_bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        **kwargs: Any,
    ) -> "VertexAIVectorStore":
        """Create VertexAIVectorStore from config."""
        return cls(
            project_id=project_id,
            region=region,
            index_name=index_id,
            endpoint_id=endpoint_id,
            gcs_bucket_name=gcs_bucket_name,
            credentials_path=credentials_path,
            text_key=text_key,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VertexAIVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._index

    @property
    def index(self) -> Any:
        """Get client."""
        return self._index

    @property
    def endpoint(self) -> Any:
        """Get client."""
        return self._endpoint

    @property
    def staging_bucket(self) -> Any:
        """Get client."""
        return self._staging_bucket

    def add(
        self,
        nodes: List[BaseNode],
        is_complete_overwrite: bool = False,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = []
        embeddings = []
        metadatas = []
        for node in nodes:
            node_id = node.node_id
            metadata = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=False
            )
            embedding = node.get_embedding()

            ids.append(node_id)
            embeddings.append(embedding)
            metadatas.append(metadata)

        data_points = utils.to_data_points(ids, embeddings, metadatas)
        # self._document_storage.add_documents(list(zip(ids, nodes)))

        if self._stream_update:
            utils.stream_update_index(index=self._index, data_points=data_points)
        else:
            if self._staging_bucket is None:
                raise ValueError(
                    "To update a Vector Search index a staging bucket must be defined."
                )
            utils.batch_update_index(
                index=self._index,
                data_points=data_points,
                staging_bucket=self._staging_bucket,
                is_complete_overwrite=is_complete_overwrite,
            )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # get datapoint ids by filter
        filter = {"ref_doc_id": ref_doc_id}
        ids = utils.get_datapoints_by_filter(
            index=self.index, endpoint=self.endpoint, metadata=filter
        )
        # remove datapoints
        self._index.remove_datapoints(datapoint_ids=ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        query_embedding = None
        if query.mode == VectorStoreQueryMode.DEFAULT:
            query_embedding = [cast(List[float], query.query_embedding)]

        if query.filters is not None:
            if "filter" in kwargs and kwargs["filter"] is not None:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for Vertex AI Vector Search specific items that are "
                    "not supported via the generic query interface such as numeric filters."
                )
            filter, num_filter = utils.to_vectorsearch_filter(query.filters)
        else:
            filter = None
            num_filter = None

        matches = utils.find_neighbors(
            index=self._index,
            endpoint=self._endpoint,
            embeddings=query_embedding,
            top_k=query.similarity_top_k,
            filter=filter,
            numeric_filter=num_filter,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for match in matches:
            node = utils.to_node(match, self.text_key)
            top_k_ids.append(match.id)
            top_k_scores.append(match.distance)
            top_k_nodes.append(node)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
