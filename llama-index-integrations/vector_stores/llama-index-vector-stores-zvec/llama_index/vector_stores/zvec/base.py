"""Zvec Vector Store."""

import logging
import json
from typing import Any, List, Dict, Optional, cast
from typing_extensions import override

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from zvec import (
    create_and_open,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Doc,
    MetricType,
    VectorQuery,
    VectorSchema,
    WeightedReRanker,
    HnswIndexParam,
)

DEFAULT_BATCH_SIZE = 100
logger = logging.getLogger(__name__)

# 定义支持的元数据类型映射
METADATA_TYPE_MAPPING = {
    "int": DataType.INT64,
    "float": DataType.FLOAT,
    "str": DataType.STRING,
    "bool": DataType.BOOL,
}


def _to_zvec_filter(
    standard_filters: Optional[MetadataFilters] = None,
) -> Optional[str]:
    """Convert from standard filter to zvec filter string."""
    if standard_filters is None:
        return None

    filters = []
    for filter_obj in standard_filters.legacy_filters():
        # 根据值类型添加引号
        if isinstance(filter_obj.value, str):
            value = f"'{filter_obj.value}'"
        else:
            value = f"{filter_obj.value}"

        filters.append(f"{filter_obj.key} = {value}")

    return " and ".join(filters)


class ZvecVectorStore(BasePydanticVectorStore):
    """
    Zvec Vector Store.

    In this vector store, embeddings and docs are stored within a
    Zvec collection.

    During query time, the index uses Zvec to query for the top
    k most similar nodes.

    Args:
        path (str): Path to the Zvec database file
        collection_name (str): Name of the Zvec collection
        collection_metadata (Optional[Dict[str, Any]]): Metadata schema definition
        embed_dim (int): Dimension of the embeddings
        support_sparse_vector (bool): Whether to support sparse vectors
        encoder (Optional[Any]): Sparse vector encoder instance

    Examples:
        `pip install llama-index-vector-stores-zvec`

        ```python
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.zvec import ZvecVectorStore

        vector_store = ZvecVectorStore(
            path="zvec_demo.zvec",
            collection_name="zvec_demo",
            embed_dim=1536
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    _metadata_keys: List[str] = PrivateAttr(default_factory=list)
    _support_sparse_vector: bool = PrivateAttr(default=False)
    _encoder: Optional[Any] = PrivateAttr(default=None)
    _collection: Optional[Collection] = PrivateAttr(default=None)

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection_metadata: Optional[Dict[str, Any]] = None,
        embed_dim: Optional[int] = None,
        support_sparse_vector: bool = False,
        encoder: Optional[Any] = None,
    ) -> None:
        """
        Initialize the ZvecVectorStore.

        Args:
            path: Path to the Zvec database file
            collection_name: Name of the Zvec collection
            collection_metadata: Metadata schema definition
            embed_dim: Embedding dimension
            support_sparse_vector: Whether to support sparse vectors
            encoder: Sparse vector encoder

        """
        super().__init__()

        # Validate required parameters
        if not path:
            raise ValueError("Path is required for ZvecVectorStore initialization")
        if not collection_name:
            raise ValueError(
                "Collection name is required for ZvecVectorStore initialization"
            )
        if not embed_dim:
            raise ValueError(
                "Embedding dimension is required for ZvecVectorStore initialization"
            )

        # Initialize attributes
        self._support_sparse_vector = support_sparse_vector
        self._encoder = encoder

        # Process metadata keys if provided
        if collection_metadata:
            self._metadata_keys = list(collection_metadata.keys())
            self._validate_metadata_types(collection_metadata)

        # Initialize sparse vector encoder if needed
        if support_sparse_vector:
            self._setup_sparse_encoder(encoder)

        # Initialize the Zvec collection
        self._collection = self._initialize_collection(
            path, collection_name, collection_metadata, embed_dim, support_sparse_vector
        )

    def _validate_metadata_types(self, collection_metadata: Dict[str, Any]) -> None:
        """Validate metadata types are supported."""
        for key, value_type in collection_metadata.items():
            if value_type not in METADATA_TYPE_MAPPING:
                raise ValueError(
                    f"Unsupported metadata type '{value_type}' for key '{key}'. "
                    f"Supported types: {list(METADATA_TYPE_MAPPING.keys())}"
                )

    def _setup_sparse_encoder(self, encoder: Optional[Any]) -> None:
        """Setup sparse vector encoder."""
        try:
            import dashtext
        except ImportError:
            raise ImportError(
                "`dashtext` package not found, please run `pip install dashtext`"
            )

        if encoder is None:
            encoder = dashtext.SparseVectorEncoder.default()

        self._encoder = cast(dashtext.SparseVectorEncoder, encoder)

    @classmethod
    def _initialize_collection(
        cls,
        path: str,
        collection_name: str,
        collection_metadata: Optional[Dict[str, Any]],
        embed_dim: int,
        support_sparse_vector: bool,
    ) -> Collection:
        """
        Initialize the Zvec collection with proper schema.

        Args:
            path: Path to the Zvec database file
            collection_name: Name of the collection
            collection_metadata: Metadata schema definition
            embed_dim: Embedding dimension
            support_sparse_vector: Whether to support sparse vectors

        Returns:
            Initialized Zvec Collection

        """
        # Define core fields required for the vector store
        fields = [
            FieldSchema(name="node_id", data_type=DataType.STRING),
            FieldSchema(name="text", data_type=DataType.STRING),
            FieldSchema(name="metadata_", data_type=DataType.STRING),
        ]

        # Add custom metadata fields if specified
        if collection_metadata:
            for key, value_type in collection_metadata.items():
                if value_type in METADATA_TYPE_MAPPING:
                    data_type = METADATA_TYPE_MAPPING[value_type]
                    fields.append(FieldSchema(name=key, data_type=data_type))
                else:
                    raise ValueError(f"Unsupported metadata type: {value_type}")

        # Define vector schemas
        vectors = [
            VectorSchema(
                name="dense_embedding",
                data_type=DataType.VECTOR_FP32,
                dimension=embed_dim,
                index_param=HnswIndexParam(metric_type=MetricType.IP),
            ),
        ]

        # Add sparse vector schema if enabled
        if support_sparse_vector:
            vectors.append(
                VectorSchema(
                    name="sparse_embedding",
                    data_type=DataType.SPARSE_VECTOR_FP32,
                    index_param=HnswIndexParam(metric_type=MetricType.IP),
                )
            )

        # Create collection schema
        collection_schema = CollectionSchema(
            name=collection_name,
            fields=fields,
            vectors=vectors,
        )

        # Create and return the collection
        return create_and_open(
            path=path,
            schema=collection_schema,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ZvecVectorStore"

    @property
    def client(self) -> Collection:
        """Return the Zvec collection."""
        if self._collection is None:
            raise RuntimeError("Collection has not been initialized")
        return self._collection

    @override
    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to vector store.

        Args:
            nodes: List of nodes with embeddings
            **add_kwargs: Additional arguments for adding nodes

        Returns:
            List of node IDs that were added

        """
        if not nodes:
            return []

        # Process nodes in batches
        all_added_ids = []
        for i in range(0, len(nodes), DEFAULT_BATCH_SIZE):
            batch_end = min(i + DEFAULT_BATCH_SIZE, len(nodes))
            batch_docs = []

            for node in nodes[i:batch_end]:
                # Prepare the document fields
                fields = self._prepare_document_fields(node)

                # Prepare vectors dictionary
                vectors_dict = {"dense_embedding": node.embedding}
                if self._support_sparse_vector:
                    vectors_dict["sparse_embedding"] = self._encoder.encode_documents(
                        node.get_content(metadata_mode=MetadataMode.EMBED)
                    )

                # Create Zvec document
                doc = Doc(
                    id=node.node_id,
                    vectors=vectors_dict,
                    fields=fields,
                )
                batch_docs.append(doc)

            # Insert the batch
            response = self._collection.insert(batch_docs)
            if not response:
                raise Exception(f"Failed to insert docs batch starting at index {i}")

            # Collect the IDs of added nodes
            batch_ids = [node.node_id for node in nodes[i:batch_end]]
            all_added_ids.extend(batch_ids)

        return all_added_ids

    def _prepare_document_fields(self, node: BaseNode) -> Dict[str, Any]:
        """
        Prepare document fields for insertion.

        Args:
            node: The node to prepare fields for

        Returns:
            Dictionary of fields for the Zvec document

        """
        # Basic fields
        fields = {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
        }

        # Process metadata
        metadata_dict = node_to_metadata_dict(
            node, remove_text=True, flat_metadata=self.flat_metadata
        )
        fields["metadata_"] = json.dumps(metadata_dict, ensure_ascii=False)

        # Add custom metadata fields if they exist in the node
        for key in self._metadata_keys:
            if key in metadata_dict:
                fields[key] = metadata_dict[key]

        return fields

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using reference document ID.

        Args:
            ref_doc_id: The document ID of the document to delete
            **delete_kwargs: Additional arguments for deletion

        """
        if not ref_doc_id:
            raise ValueError("Reference document ID cannot be empty")

        # Query for documents matching the node_id field
        filter_condition = f"node_id='{ref_doc_id}'"
        response = self._collection.query(filter=filter_condition)

        if not response:
            logger.warning(f"No documents found with filter: {filter_condition}")
            return

        # Extract IDs and delete the documents
        doc_ids = [doc.id for doc in response]
        self._collection.delete(ids=doc_ids)

    @override
    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: Vector store query object
            **kwargs: Additional query arguments

        Returns:
            Vector store query result

        """
        if not query.query_embedding and query.mode == VectorStoreQueryMode.DEFAULT:
            raise ValueError("Query embedding is required for similarity search")

        # Prepare query embedding
        query_embedding = (
            [float(e) for e in query.query_embedding] if query.query_embedding else []
        )

        # Prepare vector queries
        queries = [VectorQuery(field_name="dense_embedding", vector=query_embedding)]

        # Setup reranker for hybrid search if needed
        reranker = None
        if (
            query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            and self._support_sparse_vector
        ):
            sparse_vector = self._encoder.encode_queries(query.query_str)
            queries.append(
                VectorQuery(field_name="sparse_embedding", vector=sparse_vector)
            )

            # Calculate weights
            dense_weight = query.alpha if query.alpha is not None else 0.5
            sparse_weight = 1 - dense_weight

            # Determine top-k for hybrid search
            hybrid_top_k = query.hybrid_top_k or query.similarity_top_k

            reranker = WeightedReRanker(
                topn=hybrid_top_k,
                metric=MetricType.IP,
                weights={
                    "dense_embedding": dense_weight,
                    "sparse_embedding": sparse_weight,
                },
            )

        # Convert filters
        filter_condition = _to_zvec_filter(query.filters)

        # Execute the query
        response = self._collection.query(
            vectors=queries,
            topk=query.similarity_top_k,
            filter=filter_condition,
            include_vector=True,
            reranker=reranker,
        )

        if not response:
            logger.warning("No results returned from vector store query")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Process results
        top_k_ids = []
        top_k_nodes = []
        top_k_scores = []

        for doc in response:
            # Try to reconstruct the node from metadata
            try:
                node = metadata_dict_to_node(json.loads(doc.fields["metadata_"]))
                node.set_content(doc.fields["text"])
            except Exception as e:
                # Fallback to legacy parsing
                logger.debug(
                    f"Failed to parse Node metadata, falling back to legacy logic: {e}"
                )
                try:
                    metadata, node_info, relationships = legacy_metadata_dict_to_node(
                        json.loads(doc.fields["metadata_"])
                    )

                    text = doc.fields["text"]
                    node = TextNode(
                        id_=doc.id,
                        text=text,
                        metadata=metadata,
                        start_char_idx=node_info.get("start", None),
                        end_char_idx=node_info.get("end", None),
                        relationships=relationships,
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Both metadata parsing methods failed: {fallback_error}"
                    )
                    continue  # Skip this result

            top_k_ids.append(doc.id)
            top_k_nodes.append(node)
            top_k_scores.append(doc.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
