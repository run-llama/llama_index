"""Milvus vector store index.

An index that is built within Milvus.

"""

import logging
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
from enum import Enum


from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.milvus.utils import (
    get_default_sparse_embedding_function,
    BaseSparseEmbeddingFunction,
    ScalarMetadataFilters,
    parse_standard_filters,
    parse_scalar_filters,
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
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pymilvus import Collection, MilvusClient, DataType, AnnSearchRequest
from pymilvus.client.types import LoadState

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


class MilvusVectorStore(BasePydanticVectorStore):
    """The Milvus Vector Store.

    In this vector store we store the text, its embedding and
    a its metadata in a Milvus collection. This implementation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesn't
    exist or if `overwrite` is set to True.

    Args:
        uri (str, optional): The URI to connect to, comes in the form of
            "https://address:port" for Milvus or Zilliz Cloud service,
            or "path/to/local/milvus.db" for the lite local Milvus. Defaults to
            "./milvus_llamaindex.db".
        token (str, optional): The token for log in. Empty if not using rbac, if
            using rbac it will most likely be "username:password".
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        dim (int, optional): The dimension of the embedding vectors for the collection.
            Required if creating a new collection.
        embedding_field (str, optional): The name of the embedding field for the
            collection, defaults to DEFAULT_EMBEDDING_KEY.
        doc_id_field (str, optional): The name of the doc_id field for the collection,
            defaults to DEFAULT_DOC_ID_KEY.
        similarity_metric (str, optional): The similarity metric to use,
            currently supports IP, COSINE and L2.
        consistency_level (str, optional): Which consistency level to use for a newly
            created collection. Defaults to "Session".
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.
        text_key (str, optional): What key text is stored in in the passed collection.
            Used when bringing your own collection. Defaults to None.
        index_config (dict, optional): The configuration used for building the
            Milvus index. Defaults to None.
        search_config (dict, optional): The configuration used for searching
            the Milvus index. Note that this must be compatible with the index
            type specified by `index_config`. Defaults to None.
        collection_properties (dict, optional): The collection properties such as TTL
            (Time-To-Live) and MMAP (memory mapping). Defaults to None.
            It could include:
            - 'collection.ttl.seconds' (int): Once this property is set, data in the
                current collection expires in the specified time. Expired data in the
                collection will be cleaned up and will not be involved in searches or queries.
            - 'mmap.enabled' (bool): Whether to enable memory-mapped storage at the collection level.
        batch_size (int): Configures the number of documents processed in one
            batch when inserting data into Milvus. Defaults to DEFAULT_BATCH_SIZE.
        enable_sparse (bool): A boolean flag indicating whether to enable support
            for sparse embeddings for hybrid retrieval. Defaults to False.
        sparse_embedding_function (BaseSparseEmbeddingFunction, optional): If enable_sparse
             is True, this object should be provided to convert text to a sparse embedding.
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
        index_managemen (IndexManagement): Specifies the index management strategy to use. Defaults to "create_if_not_exists".

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
    text_key: Optional[str]
    output_fields: List[str] = Field(default_factory=list)
    index_config: Optional[dict]
    search_config: Optional[dict]
    collection_properties: Optional[dict]
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_sparse: bool = False
    sparse_embedding_field: str = "sparse_embedding"
    sparse_embedding_function: Any
    hybrid_ranker: str
    hybrid_ranker_params: dict = {}
    index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS

    _milvusclient: MilvusClient = PrivateAttr()
    _collection: Any = PrivateAttr()

    def __init__(
        self,
        uri: str = "./milvus_llamaindex.db",
        token: str = "",
        collection_name: str = "llamacollection",
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "IP",
        consistency_level: str = "Session",
        overwrite: bool = False,
        text_key: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        index_config: Optional[dict] = None,
        search_config: Optional[dict] = None,
        collection_properties: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_sparse: bool = False,
        sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None,
        hybrid_ranker: str = "RRFRanker",
        hybrid_ranker_params: dict = {},
        index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            collection_name=collection_name,
            dim=dim,
            embedding_field=embedding_field,
            doc_id_field=doc_id_field,
            consistency_level=consistency_level,
            overwrite=overwrite,
            text_key=text_key,
            output_fields=output_fields or [],
            index_config=index_config if index_config else {},
            search_config=search_config if search_config else {},
            collection_properties=collection_properties,
            batch_size=batch_size,
            enable_sparse=enable_sparse,
            sparse_embedding_function=sparse_embedding_function,
            hybrid_ranker=hybrid_ranker,
            hybrid_ranker_params=hybrid_ranker_params,
            index_management=index_management,
        )

        # Select the similarity metric
        similarity_metrics_map = {
            "ip": "IP",
            "l2": "L2",
            "euclidean": "L2",
            "cosine": "COSINE",
        }
        self.similarity_metric = similarity_metrics_map.get(
            similarity_metric.lower(), "L2"
        )
        # Connect to Milvus instance
        self._milvusclient = MilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )
        # Delete previous collection if overwriting
        if overwrite and collection_name in self.client.list_collections():
            self._milvusclient.drop_collection(collection_name)

        # Create the collection if it does not exist
        if collection_name not in self.client.list_collections():
            if dim is None:
                raise ValueError("Dim argument required for collection creation.")
            if self.enable_sparse is False:
                self._milvusclient.create_collection(
                    collection_name=collection_name,
                    dimension=dim,
                    primary_field_name=MILVUS_ID_FIELD,
                    vector_field_name=embedding_field,
                    id_type="string",
                    metric_type=self.similarity_metric,
                    max_length=65_535,
                    consistency_level=consistency_level,
                )
            else:
                try:
                    _ = DataType.SPARSE_FLOAT_VECTOR
                except Exception as e:
                    logger.error(
                        "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                    )
                    raise NotImplementedError(
                        "Hybrid retrieval requires Milvus 2.4.0 or later."
                    ) from e
                self._create_hybrid_index(collection_name)

        self._collection = Collection(collection_name, using=self._milvusclient._using)
        self._create_index_if_required()

        # Set properties
        if collection_properties:
            if self._milvusclient.get_load_state(collection_name) == LoadState.Loaded:
                self._collection.release()
                self._collection.set_properties(properties=collection_properties)
                self._collection.load()
            else:
                self._collection.set_properties(properties=collection_properties)

        self.enable_sparse = enable_sparse
        if self.enable_sparse is True and sparse_embedding_function is None:
            logger.warning("Sparse embedding function is not provided, using default.")
            self.sparse_embedding_function = get_default_sparse_embedding_function()
        elif self.enable_sparse is True and sparse_embedding_function is not None:
            self.sparse_embedding_function = sparse_embedding_function
        else:
            pass

        logger.debug(f"Successfully created a new collection: {self.collection_name}")

    @property
    def client(self) -> Any:
        """Get client."""
        return self._milvusclient

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add the embeddings and their nodes into Milvus.

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
            entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = node.embedding

            if self.enable_sparse is True:
                entry[
                    self.sparse_embedding_field
                ] = self.sparse_embedding_function.encode_documents([node.text])[0]

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        # Insert the data into milvus
        for insert_batch in iter_batch(insert_list, self.batch_size):
            self._collection.insert(insert_batch)
        if add_kwargs.get("force_flush", False):
            self._collection.flush()
        self._create_index_if_required()
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
        entries = self._milvusclient.query(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} in [{','.join(doc_ids)}]",
        )
        if len(entries) > 0:
            ids = [entry["id"] for entry in entries]
            self._milvusclient.delete(collection_name=self.collection_name, pks=ids)
            logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

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

        self._milvusclient.delete(
            collection_name=self.collection_name,
            filter=filter,
            **delete_kwargs,
        )
        logger.debug(f"Successfully deleted node_ids: {node_ids}")

    def clear(self) -> None:
        """Clears db."""
        self._milvusclient.drop_collection(self.collection_name)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes by node ids or metadata filters.

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
            if not self.text_key:
                node = metadata_dict_to_node(item)
                node.embedding = item.get(self.embedding_field, None)
            else:
                try:
                    text = item.pop(self.text_key)
                except Exception:
                    raise ValueError(
                        "The passed in text_key value does not exist "
                        "in the retrieved entity."
                    ) from None
                embedding = item.pop(self.embedding_field, None)
                node = TextNode(
                    text=text,
                    embedding=embedding,
                    metadata=item,
                )
            nodes.append(node)
        return nodes

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

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
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if self.enable_sparse is False:
                raise ValueError(f"QueryMode is HYBRID, but enable_sparse is False.")
        elif query.mode == VectorStoreQueryMode.MMR:
            pass
        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

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
        if self.text_key and self.text_key not in output_fields and outputs_limited:
            output_fields.append(self.text_key)

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = f" and ".join(expr)

        # Perform the search
        if query.mode == VectorStoreQueryMode.DEFAULT:
            # Perform default search
            res = self._milvusclient.search(
                collection_name=self.collection_name,
                data=[query.query_embedding],
                filter=string_expr,
                limit=query.similarity_top_k,
                output_fields=output_fields,
                search_params=self.search_config,
                anns_field=self.embedding_field,
            )
            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )

            nodes = []
            similarities = []
            ids = []
            # Parse the results
            for hit in res[0]:
                if not self.text_key:
                    node = metadata_dict_to_node(
                        {
                            "_node_content": hit["entity"].get("_node_content", None),
                            "_node_type": hit["entity"].get("_node_type", None),
                        }
                    )
                else:
                    try:
                        text = hit["entity"].get(self.text_key)
                    except Exception:
                        raise ValueError(
                            "The passed in text_key value does not exist "
                            "in the retrieved entity."
                        )

                    metadata = {
                        key: hit["entity"].get(key) for key in self.output_fields
                    }
                    node = TextNode(text=text, metadata=metadata)

                nodes.append(node)
                similarities.append(hit["distance"])
                ids.append(hit["id"])

        elif query.mode == VectorStoreQueryMode.MMR:
            # Perform MMR search
            mmr_threshold = kwargs.get("mmr_threshold", None)

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

            res = self._milvusclient.search(
                collection_name=self.collection_name,
                data=[query.query_embedding],
                filter=string_expr,
                limit=prefetch_k0,
                output_fields=output_fields,
                search_params=self.search_config,
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

            nodes = []
            # Parse the results
            for hit in selected_nodes:
                if not self.text_key:
                    node = metadata_dict_to_node(
                        {
                            "_node_content": hit["entity"].get("_node_content", None),
                            "_node_type": hit["entity"].get("_node_type", None),
                        }
                    )
                else:
                    try:
                        text = hit["entity"].get(self.text_key)
                    except Exception:
                        raise ValueError(
                            "The passed in text_key value does not exist "
                            "in the retrieved entity."
                        )

                    metadata = {
                        key: hit["entity"].get(key) for key in self.output_fields
                    }
                    node = TextNode(text=text, metadata=metadata)

                nodes.append(node)

            similarities = mmr_similarities  # Passing the MMR similarities instead of the original similarities
            ids = mmr_ids

            logger.debug(
                f"Successfully performed MMR on embeddings in collection: {self.collection_name}"
            )

        else:
            # Perform hybrid search
            sparse_emb = self.sparse_embedding_function.encode_queries(
                [query.query_str]
            )[0]
            sparse_search_params = {"metric_type": "IP"}

            sparse_req = AnnSearchRequest(
                data=[sparse_emb],
                anns_field=self.sparse_embedding_field,
                param=sparse_search_params,
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
            ranker = None

            if WeightedRanker is None or RRFRanker is None:
                logger.error(
                    "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                )
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

            res = self._collection.hybrid_search(
                [dense_req, sparse_req],
                rerank=ranker,
                limit=query.similarity_top_k,
                output_fields=output_fields,
            )

            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )

            nodes = []
            similarities = []
            ids = []
            # Parse the results
            for hit in res[0]:
                if not self.text_key:
                    node = metadata_dict_to_node(
                        {
                            "_node_content": hit.entity.get("_node_content"),
                            "_node_type": hit.entity.get("_node_type"),
                        }
                    )
                else:
                    try:
                        text = hit.entity.get(self.text_key)
                    except Exception:
                        raise ValueError(
                            "The passed in text_key value does not exist "
                            "in the retrieved entity."
                        )

                    metadata = {key: hit.entity.get(key) for key in self.output_fields}
                    node = TextNode(text=text, metadata=metadata)

                nodes.append(node)
                similarities.append(hit.distance)
                ids.append(hit.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _create_index_if_required(self) -> None:
        """
        Create or validate the index based on the index management strategy.

        This method decides whether to create or validate the index based on
        the specified index management strategy and the current state of the collection.
        """
        if self.index_management == IndexManagement.NO_VALIDATION:
            return

        if self.enable_sparse is False:
            self._create_dense_index()
        else:
            self._create_hybrid_index(self.collection_name)

    def _create_dense_index(self) -> None:
        """
        Create or recreate the dense vector index.

        This method handles the creation of the dense vector index based on
        the current index management strategy and the state of the collection.
        """
        index_exists = self._collection.has_index()

        if (
            not index_exists
            and self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS
        ) or (index_exists and self.overwrite):
            if index_exists:
                self._collection.release()
                self._collection.drop_index()

            base_params: Dict[str, Any] = self.index_config.copy()
            index_type: str = base_params.pop("index_type", "FLAT")
            index_params: Dict[str, Union[str, Dict[str, Any]]] = {
                "params": base_params,
                "metric_type": self.similarity_metric,
                "index_type": index_type,
            }
            self._collection.create_index(
                self.embedding_field, index_params=index_params
            )
            self._collection.load()

    def _create_hybrid_index(self, collection_name: str) -> None:
        """
        Create or recreate the hybrid (dense and sparse) vector index.

        Args:
            collection_name (str): The name of the collection to create the index for.
        """
        # Check if the collection exists, if not, create it
        if collection_name not in self._milvusclient.list_collections():
            schema = MilvusClient.create_schema(
                auto_id=False, enable_dynamic_field=True
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=65535,
                is_primary=True,
            )
            schema.add_field(
                field_name=self.embedding_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
            schema.add_field(
                field_name=self.sparse_embedding_field,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            self._milvusclient.create_collection(
                collection_name=collection_name, schema=schema
            )

        # Initialize or get the collection
        self._collection = Collection(collection_name, using=self._milvusclient._using)

        dense_index_exists = self._collection.has_index(index_name=self.embedding_field)
        sparse_index_exists = self._collection.has_index(
            index_name=self.sparse_embedding_field
        )

        if (
            (not dense_index_exists or not sparse_index_exists)
            and self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS
            or (dense_index_exists and sparse_index_exists and self.overwrite)
        ):
            if dense_index_exists:
                self._collection.release()
                self._collection.drop_index(index_name=self.embedding_field)
            if sparse_index_exists:
                self._collection.drop_index(index_name=self.sparse_embedding_field)

            # Create sparse index
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self._collection.create_index(self.sparse_embedding_field, sparse_index)

            # Create dense index
            base_params = self.index_config.copy()
            index_type = base_params.pop("index_type", "FLAT")
            dense_index = {
                "params": base_params,
                "metric_type": self.similarity_metric,
                "index_type": index_type,
            }
            self._collection.create_index(self.embedding_field, dense_index)

        self._collection.load()
