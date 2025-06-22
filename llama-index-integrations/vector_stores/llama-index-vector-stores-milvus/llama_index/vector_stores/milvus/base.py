"""
Milvus vector store index.

An index that is built within Milvus.

"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from copy import deepcopy
from enum import Enum

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.milvus.utils import (
    get_default_sparse_embedding_function,
    BaseSparseEmbeddingFunction,
    BaseMilvusBuiltInFunction,
    BM25BuiltInFunction,
    ScalarMetadataFilters,
    parse_standard_filters,
    parse_scalar_filters,
    DEFAULT_SPARSE_EMBEDDING_KEY,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pymilvus import (
    Collection,
    CollectionSchema,
    MilvusClient,
    AsyncMilvusClient,
    DataType,
    AnnSearchRequest,
)
from pymilvus.client.types import LoadState
from pymilvus.milvus_client.index import IndexParams

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
MILVUS_ID_FIELD = "id"
DEFAULT_MMR_PREFETCH_FACTOR = 4.0

try:
    from pymilvus import WeightedRanker, RRFRanker
except Exception as e:
    WeightedRanker = None
    RRFRanker = None


class IndexManagement(Enum):
    """Enumeration representing the supported index management operations."""

    NO_VALIDATION = "no_validation"
    CREATE_IF_NOT_EXISTS = "create_if_not_exists"


def _to_milvus_filter(
    standard_filters: MetadataFilters, scalar_filters: ScalarMetadataFilters = None
) -> str:
    """Translate metadata filters to Milvus specific spec."""
    standard_filters_list, joined_standard_filters = parse_standard_filters(
        standard_filters
    )
    scalar_filters_list, joined_scalar_filters = parse_scalar_filters(scalar_filters)

    filters = standard_filters_list + scalar_filters_list

    if len(standard_filters_list) > 0 and len(scalar_filters_list) > 0:
        joined_filters = f" {joined_standard_filters} and {joined_scalar_filters} "
        return f"({joined_filters})" if len(filters) > 1 else joined_filters
    elif len(standard_filters_list) > 0 and len(scalar_filters_list) == 0:
        return (
            f"({joined_standard_filters})"
            if len(filters) > 1
            else joined_standard_filters
        )
    elif len(standard_filters_list) == 0 and len(scalar_filters_list) > 0:
        return (
            f"({joined_scalar_filters})" if len(filters) > 1 else joined_scalar_filters
        )
    else:
        return ""


def _get_index_metric_type(
    func: Union[BaseSparseEmbeddingFunction, BaseMilvusBuiltInFunction],
):
    if isinstance(func, BM25BuiltInFunction):
        return "BM25"
    else:
        return "IP"


similarity_metrics_map = {
    "ip": "IP",
    "l2": "L2",
    "euclidean": "L2",
    "cosine": "COSINE",
}


class MilvusVectorStore(BasePydanticVectorStore):
    """
    The Milvus Vector Store.

    In this vector store we store the text, its embedding and
    a its metadata in a Milvus collection. This implementation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesn't
    exist or if `overwrite` is set to True.

    Args:
        uri (str): The URI to connect to, comes in the form of
            "https://address:port" for Milvus or Zilliz Cloud service,
            or "path/to/local/milvus.db" for the lite local Milvus. Defaults to
            "./milvus_llamaindex.db".
        token (str): The token for log in. Empty if not using rbac, if
            using rbac it will most likely be "username:password". Defaults to "".
        collection_name (str): The name of the collection where data will be
            stored. Defaults to "llamalection".
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.
        upsert_mode (bool, optional): Whether to upsert documents into existing collection with same node id. Defaults to False.
        doc_id_field (str, optional): The name of the doc_id field for the collection,
            defaults to DEFAULT_DOC_ID_KEY.
        text_key (str, optional): What key text is stored in in the passed collection.
            Used when bringing your own collection. Defaults to DEFAULT_TEXT_KEY.
        scalar_field_names (list, optional): The names of the extra scalar fields to be included in the collection schema.
        scalar_field_types (list, optional): The types of the extra scalar fields.
        enable_dense (bool): A boolean flag to enable or disable dense embedding. Defaults to True.
        dim (int, optional): The dimension of the embedding vectors for the collection.
            Required when creating a new collection with enable_sparse is False.
        embedding_field (str, optional): The name of the dense embedding field for the
            collection, defaults to DEFAULT_EMBEDDING_KEY.
        index_config (dict, optional): The configuration used for building the
            dense embedding index. Defaults to None.
        search_config (dict, optional): The configuration used for searching
            the Milvus dense index. Note that this must be compatible with the index
            type specified by `index_config`. Defaults to None.
        similarity_metric (str, optional): The similarity metric to use for dense embedding,
            currently supports IP, COSINE and L2.
        enable_sparse (bool): A boolean flag to enable or disable sparse embedding. Defaults to False.
        sparse_embedding_field (str): The name of sparse embedding field, defaults to DEFAULT_SPARSE_EMBEDDING_KEY.
        sparse_embedding_function (Union[BaseSparseEmbeddingFunction, BaseMilvusBuiltInFunction], optional):
            If enable_sparse is True, this object should be provided to convert text to a sparse embedding.
            Defaults to None, which uses BM25 as the default sparse embedding function,
            or BGEM3 given existing collection without built-in functions.
        sparse_index_config (dict, optional): The configuration used to build the sparse embedding index.
            Defaults to None.
        collection_properties (dict, optional): The collection properties such as TTL
            (Time-To-Live) and MMAP (memory mapping). Defaults to None.
            It could include:
            - 'collection.ttl.seconds' (int): Once this property is set, data in the
                current collection expires in the specified time. Expired data in the
                collection will be cleaned up and will not be involved in searches or queries.
            - 'mmap.enabled' (bool): Whether to enable memory-mapped storage at the collection level.
        index_management (IndexManagement): Specifies the index management strategy to use. Defaults to "create_if_not_exists".
        batch_size (int): Configures the number of documents processed in one
            batch when inserting data into Milvus. Defaults to DEFAULT_BATCH_SIZE.
        consistency_level (str, optional): Which consistency level to use for a newly
            created collection. Defaults to "Session".
        hybrid_ranker (str): Specifies the type of ranker used in hybrid search queries.
            Currently only supports ['RRFRanker','WeightedRanker']. Defaults to "RRFRanker".
        hybrid_ranker_params (dict, optional): Configuration parameters for the hybrid ranker.
            The structure of this dictionary depends on the specific ranker being used:
            - For "RRFRanker", it should include:
                - 'k' (int): A parameter used in Reciprocal Rank Fusion (RRF). This value is used
                             to calculate the rank scores as part of the RRF algorithm, which combines
                             multiple ranking strategies into a single score to improve search relevance.
            - For "WeightedRanker", it expects:
                - 'weights' (list of float): A list of exactly two weights:
                     1. The weight for the dense embedding component.
                     2. The weight for the sparse embedding component.
                  These weights are used to adjust the importance of the dense and sparse components of the embeddings
                  in the hybrid retrieval process.
            Defaults to an empty dictionary, implying that the ranker will operate with its predefined default settings.


    Raises:
        ImportError: Unable to import `pymilvus`.
        MilvusException: Error communicating with Milvus, more can be found in logging
            under Debug.

    Returns:
        MilvusVectorstore: Vectorstore that supports add, delete, and query.

    Examples:
        `pip install llama-index-vector-stores-milvus`

        ```python
        from llama_index.vector_stores.milvus import MilvusVectorStore

        # Setup MilvusVectorStore
        vector_store = MilvusVectorStore(
            dim=1536,
            collection_name="your_collection_name",
            uri="http://milvus_address:port",
            token="your_milvus_token_here",
            overwrite=True
        )
        ```

    """

    stores_text: bool = True
    stores_node: bool = True

    uri: str = "./milvus_llamaindex.db"
    token: str = ""
    collection_name: str = "llamacollection"
    dim: Optional[int]
    embedding_field: str = DEFAULT_EMBEDDING_KEY
    doc_id_field: str = DEFAULT_DOC_ID_KEY
    similarity_metric: str = "IP"
    consistency_level: str = "Session"
    overwrite: bool = False
    upsert_mode: bool = False
    text_key: str = DEFAULT_TEXT_KEY
    output_fields: List[str] = Field(default_factory=list)
    index_config: Optional[dict]
    sparse_index_config: Optional[dict]
    search_config: Optional[dict]
    collection_properties: Optional[dict]
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_dense: bool = True
    enable_sparse: bool = False
    sparse_embedding_field: str = DEFAULT_SPARSE_EMBEDDING_KEY
    sparse_embedding_function: Optional[
        Union[BaseMilvusBuiltInFunction, BaseSparseEmbeddingFunction]
    ]
    hybrid_ranker: str
    hybrid_ranker_params: dict = {}
    index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS
    scalar_field_names: Optional[List[str]]
    scalar_field_types: Optional[List[DataType]]

    _milvusclient: MilvusClient = PrivateAttr()
    _async_milvusclient: AsyncMilvusClient = PrivateAttr()
    _collection: Any = PrivateAttr()

    def __init__(
        self,
        uri: str = "./milvus_llamaindex.db",
        token: str = "",
        collection_name: str = "llamacollection",
        overwrite: bool = False,
        upsert_mode: bool = False,
        collection_properties: Optional[dict] = None,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        text_key: str = DEFAULT_TEXT_KEY,
        scalar_field_names: Optional[List[str]] = None,
        scalar_field_types: Optional[List[DataType]] = None,
        enable_dense: bool = True,
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        enable_sparse: bool = False,
        sparse_embedding_field: str = DEFAULT_SPARSE_EMBEDDING_KEY,
        sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None,
        index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        index_config: Optional[dict] = None,
        sparse_index_config: Optional[dict] = None,
        search_config: Optional[dict] = None,
        similarity_metric: str = "IP",
        consistency_level: str = "Session",
        output_fields: Optional[List[str]] = None,
        hybrid_ranker: str = "RRFRanker",
        hybrid_ranker_params: dict = {},
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            collection_name=collection_name,
            enable_dense=enable_dense,
            dim=dim,
            embedding_field=embedding_field,
            doc_id_field=doc_id_field,
            consistency_level=consistency_level,
            overwrite=overwrite,
            upsert_mode=upsert_mode,
            text_key=text_key,
            output_fields=output_fields or [],
            index_config=index_config if index_config else {},
            search_config=search_config if search_config else {},
            collection_properties=collection_properties,
            batch_size=batch_size,
            enable_sparse=enable_sparse,
            sparse_embedding_field=sparse_embedding_field,
            sparse_embedding_function=sparse_embedding_function,
            sparse_index_config=sparse_index_config if sparse_index_config else {},
            hybrid_ranker=hybrid_ranker,
            hybrid_ranker_params=hybrid_ranker_params,
            index_management=index_management,
            scalar_field_names=scalar_field_names,
            scalar_field_types=scalar_field_types,
        )
        # Connect to Milvus instance
        self._milvusclient = MilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )

        # As of writing, milvus sets alias internally in the async client.
        # This will cause an error if not removed.
        kwargs.pop("alias", None)

        self._async_milvusclient = AsyncMilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )

        # Delete previous collection if overwriting
        if overwrite and collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name)

        # Get the collection
        if collection_name in self.client.list_collections():
            self._collection = Collection(collection_name, using=self.client._using)
            self._create_index_if_required()
        else:
            self._collection = None

        # Set default args
        self.similarity_metric = similarity_metrics_map.get(
            similarity_metric.lower(), "L2"
        )
        if self.enable_dense and self.embedding_field is None:
            logger.warning("Dense embedding field name is not provided, using default.")
            self.embedding_field = DEFAULT_EMBEDDING_KEY
        if self.enable_sparse:
            if self.sparse_embedding_field is None:
                logger.warning(
                    "Sparse embedding field name is not provided, using default."
                )
                self.sparse_embedding_field = DEFAULT_SPARSE_EMBEDDING_KEY
            if self.sparse_embedding_function is None:
                logger.warning(
                    "Sparse embedding function is not provided, using default."
                )
                self.sparse_embedding_function = get_default_sparse_embedding_function(
                    input_field_names=self.text_key,
                    output_field_names=self.sparse_embedding_field,
                    collection=self._collection,
                )

        # Create the collection & index if it does not exist
        if self._collection is None:
            # Prepare schema
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema = self._add_fields_to_schema(schema)  # add fields
            schema = self._add_functions_to_schema(schema)  # add functions
            schema.verify()  # check schema

            # Prepare index
            index_params = self.client.prepare_index_params()
            if self.index_management is not IndexManagement.NO_VALIDATION:
                if self.enable_dense:
                    index_params = self._add_dense_index_params(index_params)
                if self.enable_sparse:
                    index_params = self._add_sparse_index_params(index_params)

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.debug(
                f"Successfully created a new collection: {self.collection_name}"
            )
            self._collection = Collection(collection_name, using=self.client._using)

        # Set properties
        if collection_properties:
            if self.client.get_load_state(collection_name) == LoadState.Loaded:
                self._collection.release()
                self._collection.set_properties(properties=collection_properties)
                self._collection.load()
            else:
                self._collection.set_properties(properties=collection_properties)

        logger.debug(
            f"Successfully set properties for collection: {self.collection_name}"
        )

    @property
    def client(self) -> MilvusClient:
        """Get client."""
        return self._milvusclient

    @property
    def aclient(self) -> AsyncMilvusClient:
        """Get async client."""
        return self._async_milvusclient

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add the embeddings and their nodes into Milvus.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.

        """
        insert_list = []
        insert_ids = []

        if self.enable_sparse is True and self.sparse_embedding_function is None:
            logger.fatal(
                "sparse_embedding_function is None when enable_sparse is True."
            )

        # Process that data we are going to insert
        for node in nodes:
            entry = node_to_metadata_dict(
                node, remove_text=True, text_field=self.text_key
            )
            entry[self.text_key] = node.dict()[self.text_key]
            entry[MILVUS_ID_FIELD] = node.node_id
            if self.enable_dense:
                entry[self.embedding_field] = node.embedding
            if self.enable_sparse:
                if isinstance(
                    self.sparse_embedding_function, BaseSparseEmbeddingFunction
                ):
                    entry[self.sparse_embedding_field] = (
                        self.sparse_embedding_function.encode_documents([node.text])[0]
                    )
                else:  # BaseMilvusBuiltInFunction
                    pass

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        if self.upsert_mode:
            executor_wrapper = self.client.upsert
        else:
            executor_wrapper = self.client.insert

        # Insert or Upsert the data into milvus
        for insert_batch in iter_batch(insert_list, self.batch_size):
            executor_wrapper(self.collection_name, insert_batch)
        if add_kwargs.get("force_flush", False):
            self.client.flush(self.collection_name)
        logger.debug(
            f"Successfully inserted embeddings into: {self.collection_name} "
            f"Num Inserted: {len(insert_list)}"
        )
        return insert_ids

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Asynchronous version of the add method."""
        insert_list = []
        insert_ids = []

        if self.enable_sparse is True and self.sparse_embedding_function is None:
            logger.fatal(
                "sparse_embedding_function is None when enable_sparse is True."
            )

        # Process that data we are going to insert
        for node in nodes:
            entry = node_to_metadata_dict(
                node, remove_text=True, text_field=self.text_key
            )
            entry[self.text_key] = node.dict()[self.text_key]
            entry[MILVUS_ID_FIELD] = node.node_id
            if self.enable_dense:
                entry[self.embedding_field] = node.embedding
            if self.enable_sparse:
                if isinstance(
                    self.sparse_embedding_function, BaseSparseEmbeddingFunction
                ):
                    entry[self.sparse_embedding_field] = (
                        self.sparse_embedding_function.encode_documents([node.text])[0]
                    )
                else:  # BaseMilvusBuiltInFunction
                    pass

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        if self.upsert_mode:
            executor_wrapper = self.aclient.upsert
        else:
            executor_wrapper = self.aclient.insert

        # Insert or Upsert the data into milvus
        for insert_batch in iter_batch(insert_list, self.batch_size):
            await executor_wrapper(self.collection_name, insert_batch)
        if add_kwargs.get("force_flush", False):
            raise NotImplementedError("force_flush is not supported in async mode.")
            # await self.aclient.flush(self.collection_name)
        logger.debug(
            f"Successfully inserted embeddings into: {self.collection_name} "
            f"Num Inserted: {len(insert_list)}"
        )
        return insert_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Raises:
            MilvusException: Failed to delete the doc.

        """
        # Adds ability for multiple doc delete in future.
        doc_ids: List[str]
        if isinstance(ref_doc_id, list):
            doc_ids = ref_doc_id  # type: ignore
        else:
            doc_ids = [ref_doc_id]

        # Begin by querying for the primary keys to delete
        doc_ids = ['"' + entry + '"' for entry in doc_ids]
        entries = self.client.query(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} in [{','.join(doc_ids)}]",
        )
        if len(entries) > 0:
            ids = [entry["id"] for entry in entries]
            self.client.delete(collection_name=self.collection_name, pks=ids)
            logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronous version of the delete method."""
        # Adds ability for multiple doc delete in future.
        doc_ids: List[str]
        if isinstance(ref_doc_id, list):
            doc_ids = ref_doc_id  # type: ignore
        else:
            doc_ids = [ref_doc_id]

        # Begin by querying for the primary keys to delete
        doc_ids = ['"' + entry + '"' for entry in doc_ids]
        entries = await self.aclient.query(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} in [{','.join(doc_ids)}]",
        )
        if len(entries) > 0:
            ids = [entry["id"] for entry in entries]
            await self.aclient.delete(collection_name=self.collection_name, pks=ids)
            logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        """
        filters_cpy = deepcopy(filters) or MetadataFilters(filters=[])

        if node_ids:
            filters_cpy.filters.append(
                MetadataFilter(key="id", value=node_ids, operator=FilterOperator.IN)
            )

        if filters_cpy is not None:
            filter = _to_milvus_filter(filters_cpy)
        else:
            filter = None

        self.client.delete(
            collection_name=self.collection_name,
            filter=filter,
            **delete_kwargs,
        )
        logger.debug(f"Successfully deleted node_ids: {node_ids}")

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronous version of the delete_nodes method."""
        filters_cpy = deepcopy(filters) or MetadataFilters(filters=[])

        if node_ids:
            filters_cpy.filters.append(
                MetadataFilter(key="id", value=node_ids, operator=FilterOperator.IN)
            )

        if filters_cpy is not None:
            filter = _to_milvus_filter(filters_cpy)
        else:
            filter = None

        await self.aclient.delete(
            collection_name=self.collection_name,
            filter=filter,
            **delete_kwargs,
        )
        logger.debug(f"Successfully deleted node_ids: {node_ids}")

    def clear(self) -> None:
        """Clears db."""
        self.client.drop_collection(self.collection_name)

    async def aclear(self) -> None:
        """Asynchronous version of the clear method."""
        await self.aclient.drop_collection(self.collection_name)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Get nodes by node ids or metadata filters.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to retrieve. Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters. Defaults to None.

        Raises:
            ValueError: Neither or both of node_ids and filters are provided.

        Returns:
            List[BaseNode]:

        """
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided.")

        filters_cpy = deepcopy(filters) or MetadataFilters(filters=[])
        milvus_filter = _to_milvus_filter(filters_cpy)

        if node_ids is not None and milvus_filter:
            raise ValueError("Only one of node_ids or filters can be provided.")

        res = self.client.query(
            ids=node_ids, collection_name=self.collection_name, filter=milvus_filter
        )

        nodes = []
        for item in res:
            try:
                text_content = item.get(self.text_key)
            except Exception:
                raise ValueError(
                    "The passed in text_key value does not exist "
                    "in the retrieved entity."
                )
            node = metadata_dict_to_node(item, text=text_content)
            node.embedding = item.get(self.embedding_field, None)
            nodes.append(node)
        return nodes

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronous version of the get_nodes method."""
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided.")

        filters_cpy = deepcopy(filters) or MetadataFilters(filters=[])
        milvus_filter = _to_milvus_filter(filters_cpy)

        if node_ids is not None and milvus_filter:
            raise ValueError("Only one of node_ids or filters can be provided.")

        res = await self.aclient.query(
            ids=node_ids, collection_name=self.collection_name, filter=milvus_filter
        )

        nodes = []
        for item in res:
            try:
                text_content = item.get(self.text_key)
            except Exception:
                raise ValueError(
                    "The passed in text_key value does not exist "
                    "in the retrieved entity."
                )
            node = metadata_dict_to_node(item, text=text_content)
            node.embedding = item.get(self.embedding_field, None)
            nodes.append(node)
        return nodes

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field

        """
        if query.mode == VectorStoreQueryMode.DEFAULT:
            pass
        elif query.mode in [
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            if self.enable_sparse is False:
                raise ValueError(
                    f"The query mode requires sparse embedding, but enable_sparse is False."
                )
        elif query.mode == VectorStoreQueryMode.MMR:
            pass
        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        filter_string_expr, output_fields = self._prepare_before_search(query, **kwargs)
        custom_string_expr = kwargs.pop("string_expr", "")
        if len(filter_string_expr) != 0:
            if len(custom_string_expr) != 0:
                logger.warning(
                    "string_expr in vector_store_kwargs is ignored because filters are provided."
                )
            string_expr = filter_string_expr
        else:
            string_expr = custom_string_expr

        # Perform the search
        if query.mode == VectorStoreQueryMode.MMR:
            nodes, similarities, ids = self._mmr_search(
                query, string_expr, output_fields, **kwargs
            )
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            nodes, similarities, ids = self._sparse_search(
                query, string_expr, output_fields, **kwargs
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            nodes, similarities, ids = self._hybrid_search(
                query, string_expr, output_fields
            )
        else:
            nodes, similarities, ids = self._default_search(
                query, string_expr, output_fields, **kwargs
            )
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronous version of the query method."""
        if query.mode == VectorStoreQueryMode.DEFAULT:
            pass
        elif query.mode in [
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            if self.enable_sparse is False:
                raise ValueError(
                    f"The query mode requires sparse embedding, but enable_sparse is False."
                )
        elif query.mode == VectorStoreQueryMode.MMR:
            pass
        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        string_expr, output_fields = self._prepare_before_search(query, **kwargs)

        # Perform the search
        if query.mode == VectorStoreQueryMode.MMR:
            nodes, similarities, ids = await self._async_mmr_search(
                query, string_expr, output_fields, **kwargs
            )
        elif query.mode in [
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.TEXT_SEARCH,
        ]:
            nodes, similarities, ids = await self._async_sparse_search(
                query, string_expr, output_fields, **kwargs
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            nodes, similarities, ids = await self._async_hybrid_search(
                query, string_expr, output_fields
            )
        else:
            nodes, similarities, ids = await self._async_default_search(
                query, string_expr, output_fields, **kwargs
            )
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _prepare_before_search(
        self, query: VectorStoreQuery, **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Prepare the expression and output fields for search.
        """
        expr = []
        output_fields = ["*"]
        # Parse the filter
        if query.filters is not None or "milvus_scalar_filters" in kwargs:
            expr.append(
                _to_milvus_filter(
                    query.filters,
                    (
                        kwargs["milvus_scalar_filters"]
                        if "milvus_scalar_filters" in kwargs
                        else None
                    ),
                )
            )
        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")
        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")
        # Limit output fields
        outputs_limited = False
        if query.output_fields is not None:
            output_fields = query.output_fields
            outputs_limited = True
        elif len(self.output_fields) > 0:
            output_fields = [*self.output_fields]
            outputs_limited = True
        # Add the text key to output fields if necessary
        if self.text_key not in output_fields and outputs_limited:
            output_fields.append(self.text_key)
        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = f" and ".join(expr)
        return string_expr, output_fields

    def _default_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform default search: dense embedding search.
        """
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=kwargs.get("milvus_search_config", self.search_config),
            anns_field=self.embedding_field,
        )
        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    async def _async_default_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform asynchronous default search.
        """
        res = await self.aclient.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=kwargs.get("milvus_search_config", self.search_config),
            anns_field=self.embedding_field,
        )
        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    def _mmr_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform MMR search.
        """
        mmr_threshold = kwargs.get("mmr_threshold")
        if (
            kwargs.get("mmr_prefetch_factor") is not None
            and kwargs.get("mmr_prefetch_k") is not None
        ):
            raise ValueError(
                "'mmr_prefetch_factor' and 'mmr_prefetch_k' "
                "cannot coexist in a call to query()"
            )
        else:
            if kwargs.get("mmr_prefetch_k") is not None:
                prefetch_k0 = int(kwargs["mmr_prefetch_k"])
            else:
                prefetch_k0 = int(
                    query.similarity_top_k
                    * kwargs.get("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)
                )
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=prefetch_k0,
            output_fields=output_fields,
            search_params=kwargs.get("milvus_search_config", self.search_config),
            anns_field=self.embedding_field,
        )
        nodes = res[0]
        node_embeddings = []
        node_ids = []
        for node in nodes:
            node_embeddings.append(node["entity"]["embedding"])
            node_ids.append(node["id"])
        mmr_similarities, mmr_ids = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=node_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
            mmr_threshold=mmr_threshold,
        )
        node_dict = dict(list(zip(node_ids, nodes)))
        selected_nodes = [node_dict[id] for id in mmr_ids if id in node_dict]
        similarities = mmr_similarities  # Passing the MMR similarities instead of the original similarities
        ids = mmr_ids
        nodes, _, _ = self._parse_from_milvus_results([selected_nodes])
        logger.debug(
            f"Successfully performed MMR on embeddings in collection: {self.collection_name}"
        )
        return nodes, similarities, ids

    async def _async_mmr_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform asynchronous MMR search.
        """
        mmr_threshold = kwargs.get("mmr_threshold")
        if (
            kwargs.get("mmr_prefetch_factor") is not None
            and kwargs.get("mmr_prefetch_k") is not None
        ):
            raise ValueError(
                "'mmr_prefetch_factor' and 'mmr_prefetch_k' "
                "cannot coexist in a call to query()"
            )
        else:
            if kwargs.get("mmr_prefetch_k") is not None:
                prefetch_k0 = int(kwargs["mmr_prefetch_k"])
            else:
                prefetch_k0 = int(
                    query.similarity_top_k
                    * kwargs.get("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)
                )

        res = await self.aclient.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=prefetch_k0,
            output_fields=output_fields,
            search_params=kwargs.get("milvus_search_config", self.search_config),
            anns_field=self.embedding_field,
        )
        nodes = res[0]
        node_embeddings = []
        node_ids = []
        for node in nodes:
            node_embeddings.append(node["entity"]["embedding"])
            node_ids.append(self._get_id_from_hit(node))

        mmr_similarities, mmr_ids = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=node_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
            mmr_threshold=mmr_threshold,
        )
        node_dict = dict(list(zip(node_ids, nodes)))
        selected_nodes = [node_dict[id] for id in mmr_ids if id in node_dict]
        similarities = mmr_similarities  # Passing the MMR similarities instead of the original similarities
        ids = mmr_ids
        nodes, _, _ = self._parse_from_milvus_results([selected_nodes])
        logger.debug(
            f"Successfully performed MMR on embeddings in collection: {self.collection_name}"
        )
        return nodes, similarities, ids

    def _sparse_search(
        self, query: VectorStoreQuery, string_expr: str, output_fields: List[str]
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform sparse search or full text search.
        """
        search_params = {"params": {"drop_ratio_search": 0.2}}
        if isinstance(self.sparse_embedding_function, BaseSparseEmbeddingFunction):
            sparse_emb = self.sparse_embedding_function.encode_queries(
                [query.query_str]
            )[0]
            query_data = [sparse_emb]
        elif isinstance(self.sparse_embedding_function, BaseMilvusBuiltInFunction):
            query_data = [query.query_str]
        res = self.client.search(
            collection_name=self.collection_name,
            data=query_data,
            anns_field=self.sparse_embedding_field,
            limit=query.similarity_top_k,
            filter=string_expr,
            output_fields=output_fields,
            search_params=search_params,
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    async def _async_sparse_search(
        self, query: VectorStoreQuery, string_expr: str, output_fields: List[str]
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform asynchronous sparse search.
        """
        search_params = {"params": {"drop_ratio_search": 0.2}}
        if isinstance(self.sparse_embedding_function, BaseSparseEmbeddingFunction):
            sparse_emb = self.sparse_embedding_function.encode_queries(
                [query.query_str]
            )[0]
            query_data = [sparse_emb]
        elif isinstance(self.sparse_embedding_function, BaseMilvusBuiltInFunction):
            query_data = [query.query_str]
        res = await self.aclient.search(
            collection_name=self.collection_name,
            data=query_data,
            anns_field=self.sparse_embedding_field,
            limit=query.similarity_top_k,
            filter=string_expr,
            output_fields=output_fields,
            search_params=search_params,
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    def _hybrid_search(
        self, query: VectorStoreQuery, string_expr: str, output_fields: List[str]
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform hybrid search.
        """
        if isinstance(self.sparse_embedding_function, BaseSparseEmbeddingFunction):
            sparse_emb = self.sparse_embedding_function.encode_queries(
                [query.query_str]
            )[0]
            query_data = [sparse_emb]
            sparse_metric_type = "IP"
        elif isinstance(self.sparse_embedding_function, BaseMilvusBuiltInFunction):
            query_data = [query.query_str]
            sparse_metric_type = "BM25"
        sparse_req = AnnSearchRequest(
            data=query_data,
            anns_field=self.sparse_embedding_field,
            param={"metric_type": sparse_metric_type},
            limit=query.similarity_top_k,
            expr=string_expr,  # Apply metadata filters to sparse search
        )
        dense_search_params = {
            "metric_type": self.similarity_metric,
            "params": self.search_config,
        }
        dense_emb = query.query_embedding
        dense_req = AnnSearchRequest(
            data=[dense_emb],
            anns_field=self.embedding_field,
            param=dense_search_params,
            limit=query.similarity_top_k,
            expr=string_expr,  # Apply metadata filters to dense search
        )
        if WeightedRanker is None or RRFRanker is None:
            logger.error("Hybrid retrieval is only supported in Milvus 2.4.0 or later.")
            raise ValueError(
                "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
            )
        if self.hybrid_ranker == "WeightedRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"weights": [1.0, 1.0]}
            ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
        elif self.hybrid_ranker == "RRFRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"k": 60}
            ranker = RRFRanker(self.hybrid_ranker_params["k"])
        else:
            raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")
        if not hasattr(self.client, "hybrid_search"):
            raise ValueError(
                "Your pymilvus version does not support hybrid search. please update it by `pip install -U pymilvus`"
            )
        res = self.client.hybrid_search(
            self.collection_name,
            [dense_req, sparse_req],
            ranker=ranker,
            limit=query.similarity_top_k,
            output_fields=output_fields,
        )
        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    async def _async_hybrid_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Perform asynchronous hybrid search.
        """
        if isinstance(self.sparse_embedding_function, BaseSparseEmbeddingFunction):
            sparse_emb = (
                await self.sparse_embedding_function.async_encode_queries(
                    [query.query_str]
                )
            )[0]
            query_data = [sparse_emb]
            sparse_metric_type = "IP"
        elif isinstance(self.sparse_embedding_function, BaseMilvusBuiltInFunction):
            query_data = [query.query_str]
            sparse_metric_type = "BM25"
        sparse_req = AnnSearchRequest(
            data=query_data,
            anns_field=self.sparse_embedding_field,
            param={"metric_type": sparse_metric_type},
            limit=query.similarity_top_k,
            expr=string_expr,  # Apply metadata filters to sparse search
        )
        dense_search_params = {
            "metric_type": self.similarity_metric,
            "params": self.search_config,
        }
        dense_emb = query.query_embedding
        dense_req = AnnSearchRequest(
            data=[dense_emb],
            anns_field=self.embedding_field,
            param=dense_search_params,
            limit=query.similarity_top_k,
            expr=string_expr,  # Apply metadata filters to dense search
        )
        if WeightedRanker is None or RRFRanker is None:
            logger.error("Hybrid retrieval is only supported in Milvus 2.4.0 or later.")
            raise ValueError(
                "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
            )
        if self.hybrid_ranker == "WeightedRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"weights": [1.0, 1.0]}
            ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
        elif self.hybrid_ranker == "RRFRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"k": 60}
            ranker = RRFRanker(self.hybrid_ranker_params["k"])
        else:
            raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")
        if not hasattr(self.client, "hybrid_search"):
            raise ValueError(
                "Your pymilvus version does not support hybrid search. please update it by `pip install -U pymilvus`"
            )
        res = await self.aclient.hybrid_search(
            self.collection_name,
            [dense_req, sparse_req],
            ranker=ranker,
            limit=query.similarity_top_k,
            output_fields=output_fields,
        )
        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    def _create_index_if_required(self) -> None:
        """
        Create the index based on the index management strategy.

        This method only create index for existing collection without index.
        """
        if self.index_management == IndexManagement.NO_VALIDATION:
            return
        elif self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS:
            if len(self.client.list_indexes(self.collection_name)) > 0:
                return
            else:
                index_params = self.client.prepare_index_params()
                if self.enable_dense:
                    index_params = self._add_dense_index_params(index_params)
                if self.enable_sparse:
                    index_params = self._add_sparse_index_params(index_params)
                self.client.create_index(self.collection_name, index_params)
                logger.debug(
                    f"Successfully created index for existing collection: {self.collection_name}"
                )
        else:
            logger.warning(
                f"Ignored unsupported index management strategy: {self.index_management}"
            )
            return

    def _add_dense_index_params(self, index_params: IndexParams):
        """Add dense vector index to params."""
        base_params: Dict[str, Any] = self.index_config.copy()
        field_name: str = base_params.pop("field_name", self.embedding_field)
        index_name: str = base_params.pop("index_name", self.embedding_field)
        index_type: str = base_params.pop("index_type", "FLAT")
        metric_type: str = base_params.pop("metric_type", self.similarity_metric)
        kwargs = {
            "field_name": field_name,
            "index_name": index_name,
            "index_type": index_type,
            "metric_type": metric_type,
        }
        if len(base_params) != 0:
            kwargs["params"] = base_params
        index_params.add_index(**kwargs)
        return index_params

    def _add_sparse_index_params(self, index_params: IndexParams):
        """Add sparse index params."""
        base_params: Dict[str, Any] = self.sparse_index_config.copy()
        field_name: str = base_params.pop("field_name", self.sparse_embedding_field)
        index_name: str = base_params.pop("index_name", self.sparse_embedding_field)
        index_type: str = base_params.pop("index_type", "SPARSE_INVERTED_INDEX")
        metric_type: str = base_params.pop(
            "metric_type", _get_index_metric_type(self.sparse_embedding_function)
        )
        kwargs = {
            "field_name": field_name,
            "index_name": index_name,
            "index_type": index_type,
            "metric_type": metric_type,
        }
        if len(base_params) != 0:
            kwargs["params"] = base_params
        index_params.add_index(**kwargs)
        return index_params

    def _add_fields_to_schema(self, schema: CollectionSchema):
        if self.enable_sparse and isinstance(
            self.sparse_embedding_function, BM25BuiltInFunction
        ):
            bm25_text_fields = self.sparse_embedding_function.input_field_names
            if isinstance(bm25_text_fields, str):
                bm25_text_fields = [bm25_text_fields]
        else:
            bm25_text_fields = []

        # Add scalar fields
        schema.add_field(
            field_name=MILVUS_ID_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65_535,
            is_primary=True,
        )
        schema.add_field(
            field_name=self.doc_id_field,
            datatype=DataType.VARCHAR,
            max_length=65_535,
        )
        if self.text_key in bm25_text_fields:
            schema.add_field(
                field_name=self.text_key,
                datatype=DataType.VARCHAR,
                max_length=65_535,
                **self.sparse_embedding_function.get_field_kwargs(),
            )
        else:
            schema.add_field(
                field_name=self.text_key, datatype=DataType.VARCHAR, max_length=65_535
            )
        if self.scalar_field_names is not None and self.scalar_field_types is not None:
            if len(self.scalar_field_names) != len(self.scalar_field_types):
                raise ValueError(
                    "scalar_field_names and scalar_field_types must have same length."
                )
            for field_name, field_type in zip(
                self.scalar_field_names, self.scalar_field_types
            ):
                max_length = 65_535 if field_type == DataType.VARCHAR else None
                if field_name in bm25_text_fields:
                    schema.add_field(
                        field_name=field_name,
                        datatype=field_type,
                        max_length=max_length,
                        **self.sparse_embedding_field.get_field_kwargs(),
                    )
                else:
                    schema.add_field(
                        field_name=field_name,
                        datatype=field_type,
                        max_length=max_length,
                    )

        # Add embedding field(s)
        if self.enable_dense:  # dense field
            if self.dim is None or self.embedding_field is None:
                raise ValueError(
                    "Dim and embedding_field are required to add dense embedding field."
                )
            schema.add_field(
                field_name=self.embedding_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
        if self.enable_sparse:  # sparse field
            if (
                self.sparse_embedding_function is None
                or self.sparse_embedding_field is None
            ):
                raise ValueError(
                    "Sparse embedding function and sparse_embedding_field are required to add sparse field."
                )
            schema.add_field(
                field_name=self.sparse_embedding_field,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
        return schema

    def _add_functions_to_schema(self, schema: CollectionSchema):
        if self.enable_sparse and isinstance(
            self.sparse_embedding_function, BaseMilvusBuiltInFunction
        ):
            milvus_function = self.sparse_embedding_function
            schema.add_function(milvus_function)
        return schema

    def _parse_from_milvus_results(
        self, results: List
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """
        Parses the results from Milvus and returns a list of nodes, similarities and ids.
        Only parse the first result since we are only searching for one query.
        """
        if len(results) > 1:
            logger.warning(
                "More than one result found in Milvus search. Only parsing the first result."
            )
        nodes = []
        similarities = []
        ids = []
        # Parse the results
        for hit in results[0]:
            metadata = {
                "_node_content": hit["entity"].get("_node_content", None),
                "_node_type": hit["entity"].get("_node_type", None),
            }
            for key in self.output_fields:
                metadata[key] = hit["entity"].get(key)
            node = metadata_dict_to_node(metadata)

            # Set the text field if it exists
            if self.text_key in hit["entity"]:
                text = hit["entity"].get(self.text_key)
                node.text = text

            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(self._get_id_from_hit(hit))
        return nodes, similarities, ids

    def _get_id_from_hit(self, hit: Dict) -> str:
        if "id" in hit:
            return hit["id"]
        else:
            return hit[next(iter(hit))]
