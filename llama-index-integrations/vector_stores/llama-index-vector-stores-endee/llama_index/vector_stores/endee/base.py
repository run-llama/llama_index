"""Endee Vector Store integration for LlamaIndex.

This module provides a vector store implementation that integrates LlamaIndex with
the Endee vector database, supporting both dense and hybrid (dense + sparse) search.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EF_SEARCH,
    DEFAULT_SPARSE_DIM,
    EF_CONSTRUCTION_FIELD,
    MAX_DIMENSION_ALLOWED,
    MAX_EF_SEARCH_ALLOWED,
    MAX_INDEX_NAME_LENGTH_ALLOWED,
    MAX_TOP_K_ALLOWED,
    MAX_VECTORS_PER_BATCH,
    PRECISION_VALID,
    REVERSE_OPERATOR_MAP,
    SPACE_TYPE_MAP,
    SPACE_TYPES_VALID,
    SUPPORTED_FILTER_OPERATORS,
)
from .utils import SUPPORTED_SPARSE_MODELS, get_sparse_encoder

try:
    from endee import Endee
except ImportError:
    raise ImportError(
        "Could not import endee. Please install it with `pip install endee`."
    )

logger = logging.getLogger(__name__)

class EndeeVectorStore(BasePydanticVectorStore):
    """Vector store implementation using the Endee vector database.

    EndeeVectorStore provides a high-performance vector store backed by Endee,
    supporting both dense and hybrid (dense + sparse) vector search with metadata
    filtering capabilities.

    Args:
        endee_index: Existing Endee index instance. If provided, other index
            creation parameters are ignored.
        api_token: API token for Endee service authentication. Required if
            endee_index is not provided.
        index_name: Name of the vector index. Required for new index creation.
            Must be alphanumeric with underscores, max 48 characters.
        space_type: Distance metric for vector similarity. Options: "cosine" (default),
            "l2", "ip" (inner product).
        dimension: Dimension of dense vectors. Required for new index creation.
            Max 10,000 dimensions.
        add_sparse_vector: Legacy parameter for sparse vectors (deprecated, use hybrid instead).
        text_key: Key for storing text content in metadata (default: DEFAULT_TEXT_KEY).
        batch_size: Number of vectors per batch operation (default: 100, max: 1000).
        remove_text_from_metadata: Whether to exclude text from stored metadata (default: False).
        hybrid: Enable hybrid search with dense and sparse vectors (default: False).
        sparse_dim: Dimension of sparse vectors. If > 0, hybrid mode is auto-enabled.
            Default is 30522 for BERT-based models.
        model_name: Sparse encoder model name. Options: "splade_pp" (default if None).
        precision: Vector precision type. Options: "binary", "float16",
            "float32", "int16" (default), "int8".
        M: HNSW graph connectivity parameter. Higher values improve recall but
            increase memory usage.
        ef_con: HNSW construction parameter. Higher values improve index quality
            but slow construction.

    Examples:
        >>> from llama-index import EndeeVectorStore
        >>> from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        >>>
        >>> # Create a dense-only vector store
        >>> vector_store = EndeeVectorStore.from_params(
        ...     api_token="your-api-token",
        ...     index_name="my_index",
        ...     dimension=1536,
        ...     space_type="cosine"
        ... )
        >>>
        >>> # Create a hybrid vector store with sparse encoding
        >>> hybrid_store = EndeeVectorStore.from_params(
        ...     api_token="your-api-token",
        ...     index_name="hybrid_index",
        ...     dimension=1536,
        ...     hybrid=True,
        ...     sparse_dim=30522,
        ...     model_name="splade_pp"
        ... )
        >>>
        >>> # Use with LlamaIndex
        >>> documents = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(
        ...     documents,
        ...     vector_store=vector_store
        ... )
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("What is the capital of France?")

    Note:
        For backward compatibility, the `add_sparse_vector` parameter is maintained
        but users should prefer the `hybrid` parameter for new implementations.
        When `sparse_dim > 0`, hybrid mode is automatically enabled regardless of
        the `hybrid` parameter value.
    """

    stores_text: bool = True
    flat_metadata: bool = False
    api_token: Optional[str]
    index_name: Optional[str]
    space_type: Optional[str]
    dimension: Optional[int]
    add_sparse_vector: bool
    text_key: str
    batch_size: int
    remove_text_from_metadata: bool
    hybrid: bool
    sparse_dim: Optional[int]
    model_name: Optional[str]
    precision: Optional[str]
    _endee_index: Any = PrivateAttr()
    _sparse_encoder: Optional[Callable] = PrivateAttr(default=None)

    def __init__(
        self,
        endee_index: Optional[Any] = None,
        api_token: Optional[str] = None,
        index_name: Optional[str] = None,
        space_type: Optional[str] = "cosine",
        dimension: Optional[int] = None,
        add_sparse_vector: bool = False,
        text_key: str = DEFAULT_TEXT_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        remove_text_from_metadata: bool = False,
        hybrid: bool = False,
        sparse_dim: Optional[int] = 0,
        model_name: Optional[str] = None,
        precision: Optional[str] = "int16",
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EndeeVectorStore.

        Args:
            endee_index: Existing Endee index instance (optional). If provided, other
                index creation parameters are ignored.
            api_token: API token for Endee service. Required if endee_index not provided.
            index_name: Name of the index. Required for new index creation.
            space_type: Distance metric. Options: "cosine", "l2", "ip" (default: "cosine").
            dimension: Dimension of dense vectors. Required for new index creation.
            add_sparse_vector: Legacy parameter for sparse vectors (use hybrid instead).
            text_key: Key for storing text content in metadata (default: DEFAULT_TEXT_KEY).
            batch_size: Number of vectors per batch operation (default: 100).
            remove_text_from_metadata: Whether to exclude text from stored metadata.
            hybrid: Enable hybrid search with dense and sparse vectors (default: False).
            sparse_dim: Dimension of sparse vectors. If > 0, hybrid mode is auto-enabled.
            model_name: Sparse encoder model name. Options: "splade_pp" (default if None).
            precision: Vector precision. Options: "binary", "float16", "float32",
                "int16" (default), "int8".
            M: HNSW graph connectivity parameter (optional). Higher values improve
                recall but increase memory usage.
            ef_con: HNSW construction parameter (optional). Higher values improve
                index quality but slow construction.
            **kwargs: Additional arguments passed to parent class.

        Raises:
            ValueError: If required parameters are missing or invalid.
            ImportError: If required dependencies are not installed.
        """
        try:
            super().__init__(
                index_name=index_name,
                api_token=api_token,
                space_type=space_type,
                dimension=dimension,
                sparse_dim=sparse_dim,
                hybrid=hybrid,
                model_name=model_name,
                precision=precision,
                add_sparse_vector=add_sparse_vector,
                text_key=text_key,
                batch_size=batch_size,
                remove_text_from_metadata=remove_text_from_metadata,        
            )

            # Initialize index (handles both dense and hybrid)
            if endee_index is not None:
                self._endee_index = endee_index
            else:
                # sparse_dim=None creates dense index, sparse_dim>0 creates hybrid index
                self._endee_index = self._initialize_endee_index(
                    api_token,
                    index_name,
                    dimension,
                    space_type,
                    precision,
                    sparse_dim=sparse_dim,
                    M=M,
                    ef_con=ef_con,
                )
            self._sparse_encoder = self._init_sparse_encoder(hybrid, model_name, batch_size) if hybrid else None
        except Exception as e:
            logger.error(f"Failed to initialize EndeeVectorStore: {e}")
            raise

    def _init_sparse_encoder(
        self,
        hybrid: bool,
        model_name: Optional[str],
        batch_size: int
    ) -> Optional[Callable]:
        """Initialize sparse encoder for hybrid search."""
        if model_name is None:
            model_name = "splade_pp"
            logger.info(f"Using default sparse model: {model_name}")
        elif model_name not in SUPPORTED_SPARSE_MODELS:
            logger.warning(
                f"Unsupported sparse model '{model_name}' provided. "
                "Falling back to default 'splade_pp'. "
                f"Supported models: {list(SUPPORTED_SPARSE_MODELS.keys())}"
            )
            model_name = "splade_pp"

        logger.info(f"Initializing sparse encoder with model: {model_name}")
        try:
            return get_sparse_encoder(
                model_name=model_name,
                use_fastembed=True,
                batch_size=batch_size,
            )
        except ImportError as import_err:
            logger.error(
                f"Failed to initialize sparse encoder: {import_err}. "
                "Please install fastembed: pip install endee-llamaindex[hybrid]"
            )
            raise

    @classmethod
    def _validate_index_params(
        cls,
        index_name: Optional[str],
        dimension: int,
        space_type: str,
        precision: str,
        sparse_dim: int,
    ) -> None:
        """Validate all index creation parameters."""
        try:
            from endee.utils import is_valid_index_name
            if not is_valid_index_name(index_name):
                raise ValueError(
                    f"Invalid index name. Index name must be alphanumeric and can "
                    f"contain underscores and should be less than "
                    f"{MAX_INDEX_NAME_LENGTH_ALLOWED} characters"
                )
        except ImportError:
            pass

        if dimension > MAX_DIMENSION_ALLOWED:
            raise ValueError(f"Dimension cannot be greater than {MAX_DIMENSION_ALLOWED}")

        if sparse_dim < 0:
            raise ValueError("sparse_dim cannot be negative")

        if space_type and space_type.lower() not in SPACE_TYPES_VALID:
            raise ValueError(f"Invalid space type: {space_type}. Use one of {SPACE_TYPES_VALID}")

        if precision not in PRECISION_VALID:
            raise ValueError(f"Invalid precision: {precision}. Use one of {PRECISION_VALID}")

    @classmethod
    def _initialize_endee_index(
        cls,
        api_token: Optional[str],
        index_name: Optional[str],
        dimension: Optional[int] = None,
        space_type: Optional[str] = "cosine",
        precision: Optional[str] = "int16",
        sparse_dim: Optional[int] = 0,
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
    ) -> Any:
        """Initialize or retrieve an Endee index.

        This method attempts to retrieve an existing index by name. If the index
        doesn't exist, it creates a new one with the specified parameters.

        Args:
            api_token: API token for Endee service authentication.
            index_name: Name of the index to create or retrieve.
            dimension: Dense vector dimension. Required for new index creation.
            space_type: Distance metric. Options: "cosine", "l2", "ip".
            precision: Vector precision type. Options: "binary", "float16", "float32",
                "int16" (default), "int8".
            sparse_dim: Sparse vector dimension. If 0 or None, creates dense-only index.
                If > 0, creates hybrid index with both dense and sparse vectors.
            M: HNSW M parameter (bi-directional links per node). Controls graph connectivity.
            ef_con: HNSW ef_construction parameter. Controls index build quality vs speed.

        Returns:
            Endee Index object (either retrieved or newly created).

        Raises:
            ValueError: If required parameters are missing or invalid (e.g., dimension > 10000,
                invalid index name, invalid space type, invalid precision).
            Exception: If connection to Endee service fails or index creation fails.
        """
        try:
            logger.info("Connecting to Endee service...")
            nd = Endee(token=api_token)
            is_hybrid = sparse_dim > 0

            try:
                logger.info(f"Checking if index '{index_name}' exists...")
                index = nd.get_index(name=index_name)
                existing_sparse_dim = getattr(index, "sparse_dim", 0)

                if is_hybrid and existing_sparse_dim > 0:
                    logger.info(f"Retrieved existing hybrid index: {index_name}")
                elif not is_hybrid and existing_sparse_dim == 0:
                    logger.info(f"Retrieved existing dense index: {index_name}")
                elif is_hybrid and existing_sparse_dim == 0:
                    logger.warning(
                        f"Index '{index_name}' exists as dense-only (sparse_dim=0) but hybrid was requested. "
                        f"Using existing dense index."
                    )
                else:
                    logger.warning(
                        f"Index '{index_name}' exists as hybrid (sparse_dim={existing_sparse_dim}) "
                        f"but dense-only was requested. Using existing hybrid index."
                    )
                return index

            except Exception as e:
                if dimension is None:
                    raise ValueError(
                        f"Must provide dimension when creating a new {'hybrid' if is_hybrid else 'dense'} index"
                    ) from e
                if is_hybrid and sparse_dim == 0:
                    raise ValueError("Must provide sparse_dim when creating a new hybrid index") from e

                cls._validate_index_params(index_name, dimension, space_type, precision, sparse_dim)

                create_kwargs = {
                    "name": index_name,
                    "dimension": dimension,
                    "space_type": space_type,
                    "precision": precision,
                    "sparse_dim": sparse_dim,
                }
                if M is not None:
                    create_kwargs["M"] = M
                if ef_con is not None:
                    create_kwargs["ef_con"] = ef_con

                index_type = "hybrid" if is_hybrid else "dense"
                logger.info(f"Creating new {index_type} index '{index_name}' (dimension={dimension}, sparse_dim={sparse_dim})")

                nd.create_index(**create_kwargs)
                logger.info("Index created successfully")
                return nd.get_index(name=index_name)

        except Exception as e:
            logger.error(f"Error initializing Endee index: {e}")
            raise

    @classmethod
    def from_params(
        cls,
        api_token: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        space_type: str = "cosine",
        batch_size: int = DEFAULT_BATCH_SIZE,
        hybrid: bool = False,
        sparse_dim: Optional[int] = 0,
        model_name: Optional[str] = None,
        precision: Optional[str] = "int16",
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
    ) -> "EndeeVectorStore":
        """Create EndeeVectorStore from configuration parameters.

        This is the recommended factory method for creating an EndeeVectorStore instance.
        It handles index initialization and configuration validation automatically.

        Args:
            api_token: API token for Endee service authentication. Required.
            index_name: Name of the index to create or connect to. Required.
            dimension: Dense vector dimension. Required for new index creation.
            space_type: Distance metric. Options: "cosine" (default), "l2", "ip".
            batch_size: Number of vectors per batch operation (default: 100, max: 1000).
            hybrid: Enable hybrid search mode (default: False). Auto-enabled if sparse_dim > 0.
            sparse_dim: Sparse vector dimension for hybrid search. If > 0, enables hybrid mode
                automatically and uses DEFAULT_SPARSE_DIM (30522) for BERT-based models.
            model_name: Sparse encoder model for hybrid search. Options: "splade_pp" (default).
                Only used when hybrid=True or sparse_dim > 0.
            precision: Vector precision type. Options: "binary", "float16",
                "float32", "int16" (default), "int8".
            M: HNSW M parameter controlling graph connectivity. Higher values improve recall
                but increase memory. If not provided, uses backend default.
            ef_con: HNSW ef_construction parameter controlling build quality. Higher values
                improve index quality but slow construction. If not provided, uses backend default.

        Returns:
            EndeeVectorStore: Configured vector store instance ready for use.

        Raises:
            ValueError: If required parameters are missing or invalid.
            ImportError: If required dependencies (endee, fastembed) are not installed.

        Examples:
            >>> # Dense-only vector store
            >>> store = EndeeVectorStore.from_params(
            ...     api_token="your-token",
            ...     index_name="my_index",
            ...     dimension=1536
            ... )
            >>>
            >>> # Hybrid vector store with custom model
            >>> store = EndeeVectorStore.from_params(
            ...     api_token="your-token",
            ...     index_name="hybrid_index",
            ...     dimension=1536,
            ...     hybrid=True,
            ...     sparse_dim=30522,
            ...     model_name="splade_pp"
            ... )
        """
        # Auto-enable hybrid if sparse_dim is provided and > 0
        try:
            if sparse_dim > 0:
                logger.info(f"Auto-enabling hybrid mode (sparse_dim={sparse_dim} > 0)")
                sparse_dim = DEFAULT_SPARSE_DIM

            # Initialize index (unified method handles both dense and hybrid)
            endee_index = cls._initialize_endee_index(
                api_token,
                index_name,
                dimension,
                space_type,
                precision,
                sparse_dim=sparse_dim,
                M=M,
                ef_con=ef_con,
            )

            # Get actual index configuration from backend
            try:
                index_info = endee_index.describe()
                actual_index_name = index_info.get("name", index_name)
                actual_dimension = index_info.get("dimension", dimension)
                actual_space_type = index_info.get("space_type", space_type)
                actual_precision = index_info.get("precision", precision)
                actual_sparse_dim = index_info.get("sparse_dim", sparse_dim)
            except Exception as e:
                logger.warning(f"Could not get index info, using provided parameters: {e}")
                actual_index_name, actual_dimension, actual_space_type = index_name, dimension, space_type
                actual_precision, actual_sparse_dim = precision, sparse_dim

            actual_hybrid = actual_sparse_dim > 0

            return cls(
                endee_index=endee_index,
                api_token=api_token,
                index_name=actual_index_name,
                dimension=actual_dimension,
                space_type=actual_space_type,
                batch_size=batch_size,
                sparse_dim=actual_sparse_dim,
                hybrid=actual_hybrid,
                model_name=model_name,
                precision=actual_precision,
                M=M,
                ef_con=ef_con,
            )
        except Exception as e:
            logger.error(f"Error creating EndeeVectorStore from params: {e}")
            raise

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "EndeeVectorStore"

    def _compute_sparse_vectors(self, texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """Compute sparse vectors for a batch of texts.

        Uses the configured sparse encoder (e.g., SPLADE) to generate sparse
        representations suitable for hybrid search.

        Args:
            texts: List of text strings to encode into sparse vectors.

        Returns:
            Tuple containing:
                - indices: List of lists, where each inner list contains the non-zero
                    indices for a single text's sparse vector.
                - values: List of lists, where each inner list contains the corresponding
                    values for the non-zero indices.

        Raises:
            ValueError: If sparse encoder is not initialized (hybrid mode not enabled).
            Exception: If encoding fails due to model or input errors.
        """
        if self._sparse_encoder is None:
            raise ValueError(
                "Sparse encoder is not initialized. Hybrid mode requires a sparse "
                "encoder. Please create the store with hybrid=True and provide a "
                "valid model_name (e.g., 'splade_pp')."
            )
        try:
            return self._sparse_encoder(texts)
        except Exception as e:
            logger.error(f"Failed to compute sparse vectors: {e}")
            raise RuntimeError(
                f"Sparse vector encoding failed: {e}. Check that the model is "
                "properly installed and texts are valid."
            ) from e

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector index with automatic batching.

        Performs batch upsert operations with automatic deduplication. For hybrid
        indexes, sparse vectors are automatically computed from node text content
        using the configured sparse encoder.

        Args:
            nodes: List of BaseNode objects to add. Each node must contain:
                - node_id: Unique identifier (duplicates within batch are deduplicated)
                - embedding: Dense vector representation (required)
                - text content: Text string (required for hybrid mode sparse encoding)
                - metadata: Optional dictionary of key-value pairs for filtering
            **add_kwargs: Additional keyword arguments for interface compatibility (unused).

        Returns:
            List[str]: List of node IDs that were successfully added to the index.

        Raises:
            ValueError: If nodes are missing required fields (embedding, text for hybrid mode).
            RuntimeError: If batch upsert operation fails or sparse encoding fails.

        Note:
            - Duplicate node IDs within a batch are automatically deduplicated (last occurrence wins).
            - Metadata fields (ref_doc_id, file_name, category, etc.) are extracted for query filtering.
            - Batch size is capped at MAX_VECTORS_PER_BATCH (1000) per API constraints.
            - For hybrid mode, sparse vectors are computed in batches for efficiency.
        """
        try:
            # Determine if hybrid mode is enabled
            use_hybrid = self.hybrid

            # Deduplicate nodes by ID (keep last occurrence)
            # Endee rejects duplicate IDs within a single batch
            seen: Dict[str, int] = {}
            for idx, node in enumerate(nodes):
                seen[node.node_id] = idx
            nodes = [nodes[i] for i in sorted(seen.values())]

            ids = []
            entries = []

            # Collect texts for sparse encoding if hybrid mode
            if use_hybrid:
                texts = [node.get_content() for node in nodes]

                # Compute sparse vectors in batch
                if self._sparse_encoder is not None and texts:
                    sparse_indices, sparse_values = self._compute_sparse_vectors(texts)
                else:
                    sparse_indices = [[] for _ in texts]
                    sparse_values = [[] for _ in texts]

            for i, node in enumerate(nodes):
                node_id = node.node_id
                metadata = node_to_metadata_dict(node)

                # Extract filterable metadata fields
                filter_data = {}
                ref_id = getattr(node, "ref_doc_id", None) or metadata.get("ref_doc_id")
                if ref_id is not None:
                    filter_data["ref_doc_id"] = ref_id
                for field in ["file_name", "doc_id", "category", "difficulty", "language", "field", "type", "feature"]:
                    if field in metadata:
                        filter_data[field] = metadata[field]

                # Build upsert entry
                entry = {
                    "id": node_id,
                    "vector": node.get_embedding(),
                    "meta": metadata,
                    "filter": filter_data,
                }
                if use_hybrid:
                    entry["sparse_indices"] = sparse_indices[i]
                    entry["sparse_values"] = sparse_values[i]

                ids.append(node_id)
                entries.append(entry)

            # Perform batch upsert with size limited by API constraints
            batch_size = min(self.batch_size, MAX_VECTORS_PER_BATCH)
            for i in range(0, len(entries), batch_size):
                batch = entries[i : i + batch_size]
                self._endee_index.upsert(batch)

            return ids
        except Exception as e:
            logger.error(f"Error adding nodes to index: {e}")
            raise

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes associated with a reference document ID.

        Uses metadata filtering to remove all vectors that match the specified
        ref_doc_id in their filter metadata. This enables deletion of all chunks
        belonging to a specific document.

        Args:
            ref_doc_id: Reference document ID to delete. All nodes with this
                ref_doc_id in their metadata will be removed.
            **delete_kwargs: Additional keyword arguments for interface compatibility (unused).

        Raises:
            RuntimeError: If deletion operation fails due to connection or API errors.

        Examples:
            >>> store.delete("doc_123")  # Deletes all nodes from document "doc_123"
        """
        try:
            # Filter format consistent with query: list of {field: {$op: value}}
            filter_dict = [{"ref_doc_id": {"$eq": ref_doc_id}}]
            self._endee_index.delete_with_filter(filter_dict)
        except Exception as e:
            logger.error(f"Failed to delete nodes with ref_doc_id '{ref_doc_id}': {e}")
            raise RuntimeError(
                f"Deletion failed for ref_doc_id '{ref_doc_id}': {e}"
            ) from e

    @property
    def client(self) -> Any:
        """Get the underlying Endee index client.

        Returns:
            The Endee Index object used for direct API access.
        """
        return self._endee_index

    def describe(self) -> Dict[str, Any]:
        """Get index metadata and configuration details.

        Retrieves comprehensive information about the index including dimensions,
        distance metric, precision, and other configuration parameters.

        Returns:
            Dict[str, Any]: Dictionary containing index metadata:
                - name (str): Index name
                - dimension (int): Dense vector dimension
                - space_type (str): Distance metric (cosine, l2, or ip)
                - precision (str): Vector precision type
                - sparse_dim (int): Sparse vector dimension (if hybrid index)
                - ef_con (int): HNSW ef_construction parameter
                - Additional backend-specific metadata and statistics

            Returns empty dict if the describe operation fails.

        Examples:
            >>> info = store.describe()
            >>> print(f"Index: {info['name']}, Dimension: {info['dimension']}")
        """
        try:
            return self._endee_index.describe()
        except Exception as e:
            logger.error(f"Failed to describe index: {e}")
            return {}

    def fetch(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch vectors by their IDs.

        Retrieves complete vector data including embeddings and metadata for
        the specified node IDs. This is useful for debugging or manual inspection.

        Args:
            ids: List of node IDs (vector IDs) to fetch from the index.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, one per successfully fetched vector.
                Each dictionary contains:
                    - id: The node ID
                    - vector: The dense embedding
                    - meta: The metadata dictionary
                    - filter: The filterable metadata
                    - sparse_indices/sparse_values: Sparse components (if hybrid)

            Note: Failed fetches are logged but do not raise exceptions. Only
            successfully fetched vectors are included in the result.

        Examples:
            >>> vectors = store.fetch(["node_1", "node_2", "node_3"])
            >>> for vec in vectors:
            ...     print(f"ID: {vec['id']}, Embedding dim: {len(vec['vector'])}")
        """
        results: List[Dict[str, Any]] = []
        for vector_id in ids:
            try:
                vector_data = self._endee_index.get_vector(vector_id)
                results.append(vector_data)
            except Exception as e:
                logger.warning(f"Failed to fetch vector with ID '{vector_id}': {e}")
                # Continue fetching other vectors despite individual failures
        return results

    def update_filters(self, updates: List[Dict[str, Any]]) -> str:
        """Update filter metadata for multiple vectors by ID.

        Delegates to the underlying Endee Index's ``update_filters()`` method,
        which replaces the stored filter dict for each vector in a single API call.

        Args:
            updates: List of dicts, each containing:
                - ``id`` (str): Vector identifier.
                - ``filter`` (dict): New filter metadata to replace the existing one.

        Returns:
            str: Success message returned by the Endee API (e.g.
                "Successfully updated filters for N vectors").

        Raises:
            RuntimeError: If the underlying API call fails.

        Examples:
            >>> store.update_filters([
            ...     {"id": "node_1", "filter": {"category": "ai", "status": "reviewed"}},
            ...     {"id": "node_2", "filter": {"category": "database"}},
            ... ])
        """
        try:
            return self._endee_index.update_filters(updates)
        except Exception as e:
            logger.error(f"Failed to update filters: {e}")
            raise RuntimeError(f"update_filters failed: {e}") from e

    def _get_dimension(self, query: VectorStoreQuery) -> int:
        """Determine vector dimension from index or query embedding.

        Args:
            query: VectorStoreQuery containing optional query_embedding.

        Returns:
            int: Vector dimension.

        Raises:
            ValueError: If dimension cannot be determined.
        """
        if hasattr(self._endee_index, "dimension"):
            return self._endee_index.dimension

        try:
            return self._endee_index.describe()["dimension"]
        except Exception as e:
            logger.warning(f"Could not retrieve dimension from index metadata: {e}")

        if query.query_embedding is not None:
            logger.debug(f"Inferred dimension {len(query.query_embedding)} from query embedding")
            return len(query.query_embedding)

        raise ValueError(
            "Cannot determine vector dimension. Index metadata unavailable "
            "and query_embedding not provided."
        )

    def _process_filters(self, query: VectorStoreQuery) -> Optional[List[Dict[str, Any]]]:
        """Process and validate metadata filters from query.

        Args:
            query: VectorStoreQuery containing optional filters.

        Returns:
            Optional list of filter dicts in Endee API format, or None if no filters.

        Raises:
            ValueError: If filters use unsupported operators or invalid format.
        """
        if query.filters is None:
            return None

        filters: Dict[str, Dict[str, Any]] = {}

        for filter_item in query.filters.filters:
            # Case 1: MetadataFilter object with key, value, and operator
            if (
                hasattr(filter_item, "key")
                and hasattr(filter_item, "value")
                and hasattr(filter_item, "operator")
            ):
                if filter_item.operator not in SUPPORTED_FILTER_OPERATORS:
                    raise ValueError(
                        f"Unsupported filter operator: {filter_item.operator}. "
                        f"Supported operators: {', '.join(str(op) for op in SUPPORTED_FILTER_OPERATORS)}"
                    )
                op_symbol = REVERSE_OPERATOR_MAP[filter_item.operator]
                if filter_item.key not in filters:
                    filters[filter_item.key] = {}
                filters[filter_item.key][op_symbol] = filter_item.value

            # Case 2: Raw dict format, e.g. {"category": {"$eq": "programming"}}
            elif isinstance(filter_item, dict):
                for key, op_dict in filter_item.items():
                    if isinstance(op_dict, dict):
                        for op, val in op_dict.items():
                            if key not in filters:
                                filters[key] = {}
                            filters[key][op] = val
            else:
                raise ValueError(
                    f"Unsupported filter format: {type(filter_item).__name__}. "
                    "Expected MetadataFilter object or dict."
                )

        if not filters:
            return None

        logger.debug(f"Parsed filters: {filters}")
        filter_for_api = [{field: ops} for field, ops in filters.items()]
        logger.debug(f"Filters formatted for API: {filter_for_api}")
        return filter_for_api

    def _prepare_query_embedding(
        self,
        query: VectorStoreQuery,
        dimension: int,
        use_hybrid: bool
    ) -> List[float]:
        """Prepare query embedding with optional alpha weighting.

        Args:
            query: VectorStoreQuery containing query_embedding and alpha.
            dimension: Vector dimension for zero vector initialization.
            use_hybrid: Whether hybrid mode is enabled.

        Returns:
            Query embedding vector (zero vector if not provided).
        """
        if query.query_embedding is None:
            return [0.0] * dimension

        query_embedding = cast(List[float], query.query_embedding)

        # Apply alpha weighting in hybrid mode
        if query.alpha is not None and use_hybrid:
            query_embedding = [v * query.alpha for v in query_embedding]
            logger.debug(f"Applied alpha weighting ({query.alpha}) to dense query vector")

        return query_embedding

    def _process_sparse_query(
        self,
        query: VectorStoreQuery
    ) -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """Process sparse query components for hybrid search.

        Args:
            query: VectorStoreQuery containing optional query_str.

        Returns:
            Tuple of (sparse_indices, sparse_values) or (None, None) if not applicable.
        """
        query_text = getattr(query, "query_str", None)

        if not query_text:
            logger.warning(
                "Hybrid mode enabled but no query_str provided in VectorStoreQuery. "
                "Sparse component will be empty (dense-only search)."
            )
            return None, None

        if self._sparse_encoder is None:
            logger.warning(
                "Hybrid mode enabled but sparse encoder is not initialized. "
                "Query will fall back to dense-only search."
            )
            return None, None

        logger.debug(f"Computing sparse vectors for query text: '{query_text[:100]}...'")
        si, sv = self._compute_sparse_vectors([query_text])
        sparse_indices_q = si[0]
        sparse_values_q = [float(v) for v in sv[0]]
        logger.debug(f"Generated sparse vector with {len(sparse_indices_q)} non-zero features")

        return sparse_indices_q, sparse_values_q

    def _build_query_params(
        self,
        query_embedding: List[float],
        top_k: int,
        ef: int,
        filter_for_api: Optional[List[Dict[str, Any]]],
        sparse_indices: Optional[List[int]],
        sparse_values: Optional[List[float]],
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build query parameters dict for Endee API.

        Args:
            query_embedding: Dense query vector.
            top_k: Number of results to return (already capped).
            ef: HNSW ef_search parameter (already capped).
            filter_for_api: Optional filters in API format.
            sparse_indices: Optional sparse vector indices.
            sparse_values: Optional sparse vector values.
            prefilter_cardinality_threshold: Controls when search switches from
                HNSW filtered search to brute-force prefiltering. Range: 1,000–1,000,000.
            filter_boost_percentage: Expands the HNSW candidate pool by this
                percentage when a filter is active. Range: 0–100.

        Returns:
            Dict of query parameters ready for Endee API.
        """
        query_kwargs: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "ef": ef,
            "include_vectors": True,
        }

        if filter_for_api is not None:
            query_kwargs["filter"] = filter_for_api
        if sparse_indices is not None:
            query_kwargs["sparse_indices"] = sparse_indices
        if sparse_values is not None:
            query_kwargs["sparse_values"] = sparse_values
        if prefilter_cardinality_threshold is not None:
            query_kwargs["prefilter_cardinality_threshold"] = prefilter_cardinality_threshold
        if filter_boost_percentage is not None:
            query_kwargs["filter_boost_percentage"] = filter_boost_percentage

        return query_kwargs

    def _create_node_from_legacy_metadata(
        self,
        metadata: Dict[str, Any],
        node_id: str
    ) -> BaseNode:
        """Create node from legacy metadata format with _node_content."""
        metadata_dict, node_info, relationships = legacy_metadata_dict_to_node(
            metadata=metadata,
            text_key=self.text_key,
        )

        _node_content_str = metadata.get("_node_content", "{}")
        try:
            node_content = json.loads(_node_content_str)
        except json.JSONDecodeError as json_err:
            logger.warning(f"Failed to parse _node_content for node {node_id}: {json_err}")
            node_content = {}

        node = TextNode(
            text=node_content.get(self.text_key, ""),
            metadata=metadata_dict,
            relationships=relationships,
            id_=node_id,
        )

        for key, val in node_info.items():
            if hasattr(node, key):
                setattr(node, key, val)

        return node

    def _process_single_result(self, result: Dict[str, Any]) -> Tuple[BaseNode, float, str]:
        """Process a single query result into a LlamaIndex node.

        Args:
            result: Raw result dict from Endee API.

        Returns:
            Tuple of (node, similarity_score, node_id).

        Raises:
            Exception: If result processing fails.
        """
        node_id = result["id"]
        score = result.get("similarity", result.get("score", 0.0))
        metadata = result.get("meta", {})

        node = (
            metadata_dict_to_node(metadata=metadata, text=metadata.pop(self.text_key, None), id_=node_id)
            if self.flat_metadata
            else self._create_node_from_legacy_metadata(metadata, node_id)
        )

        if "vector" in result:
            node.embedding = result["vector"]

        return node, score, node_id

    def _process_query_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """Convert Endee API results to LlamaIndex format.

        Args:
            results: List of result dicts from Endee API.

        Returns:
            Tuple of (nodes, similarities, ids).
        """
        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for result in results:
            try:
                node, score, node_id = self._process_single_result(result)
                nodes.append(node)
                similarities.append(score)
                ids.append(node_id)
            except Exception as result_error:
                logger.warning(
                    f"Failed to process result for node {result.get('id', 'unknown')}: "
                    f"{result_error}. Skipping this result."
                )
                continue

        logger.debug(f"Successfully processed {len(nodes)} nodes from query results")
        return nodes, similarities, ids

    def query(
        self,
        query: VectorStoreQuery,
        ef: int = DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[int] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the index for the top-k most similar nodes.

        Performs dense-only or hybrid (dense + sparse) vector search depending on
        index configuration. Supports metadata filtering, HNSW tuning, and alpha
        weighting for hybrid queries.

        Args:
            query: VectorStoreQuery object containing query parameters:
                - query_embedding (List[float], optional): Dense query vector. Required
                    for dense searches. If not provided, zero vector is used.
                - query_str (str, optional): Text query string. Required for hybrid
                    search sparse encoding. Ignored in dense-only mode.
                - similarity_top_k (int, optional): Number of results to return
                    (default: 10, max: 512).
                - filters (MetadataFilters, optional): Metadata filters for result
                    filtering. Supports FilterOperator.EQ and FilterOperator.IN.
                - alpha (float, optional): Hybrid search weighting factor (0.0 to 1.0).
                    - 0.0 = sparse-only (keyword search)
                    - 1.0 = dense-only (semantic search)
                    - 0.5 = balanced hybrid search (default)
                    Only used in hybrid mode.
            ef: HNSW ef_search parameter (default: 128, max: 1024). Controls the
                size of the dynamic candidate list during search. Higher values
                improve recall at the cost of speed.
            prefilter_cardinality_threshold: Controls when the search strategy
                switches from HNSW filtered search to brute-force prefiltering on
                the matched subset (default: None, uses server default of 10,000).
                Range: 1,000–1,000,000. Lower values favor HNSW graph search;
                higher values cause prefiltering to kick in more often.
            filter_boost_percentage: Expands the internal HNSW candidate pool by
                this percentage when a filter is active, compensating for
                filtered-out results (default: None, uses server default of 0).
                Range: 0–100. Set to e.g. 30 to fetch 30% more candidates before
                applying the filter.
            **kwargs: Additional keyword arguments for interface compatibility (unused).

        Returns:
            VectorStoreQueryResult: Query results containing:
                - nodes (List[BaseNode]): List of matching nodes with metadata,
                    embeddings, and relationships.
                - similarities (List[float]): Similarity scores for each node.
                    Higher scores indicate better matches.
                - ids (List[str]): Node IDs corresponding to the returned nodes.

        Raises:
            ValueError: If required query parameters are missing, filters use
                unsupported operators, or dimension cannot be determined.
            RuntimeError: If the query execution fails due to API or connection errors.

        Examples:
            >>> # Dense-only search
            >>> query = VectorStoreQuery(
            ...     query_embedding=[0.1, 0.2, ...],  # 1536-dim vector
            ...     similarity_top_k=10
            ... )
            >>> result = store.query(query)
            >>>
            >>> # Hybrid search with filtering
            >>> from llama_index.core.vector_stores.types import (
            ...     MetadataFilters, FilterOperator, FilterCondition
            ... )
            >>> filters = MetadataFilters(
            ...     filters=[
            ...         MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
            ...     ],
            ...     condition=FilterCondition.AND
            ... )
            >>> query = VectorStoreQuery(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     query_str="machine learning algorithms",
            ...     similarity_top_k=5,
            ...     filters=filters,
            ...     alpha=0.7  # Favor dense search
            ... )
            >>> result = store.query(query, ef=256)  # Higher recall

        Note:
            - For hybrid indexes, providing query_str enables sparse vector computation
            - Filters are applied at query time, not post-processing
            - ef parameter is capped at MAX_EF_SEARCH_ALLOWED (1024)
            - top_k is capped at MAX_TOP_K_ALLOWED (512)
        """
        try:
            use_hybrid = self.hybrid
            logger.debug(
                f"Query mode: {'hybrid (dense + sparse)' if use_hybrid else 'dense-only'}"
            )

            # Get vector dimension
            dimension = self._get_dimension(query)

            # Process metadata filters
            filter_for_api = self._process_filters(query)

            # Prepare query embedding with optional alpha weighting
            query_embedding = self._prepare_query_embedding(query, dimension, use_hybrid)

            # Process sparse components for hybrid search
            sparse_indices_q: Optional[List[int]] = None
            sparse_values_q: Optional[List[float]] = None
            if use_hybrid:
                sparse_indices_q, sparse_values_q = self._process_sparse_query(query)
            else:
                logger.debug("Dense-only search mode - sparse encoding skipped")

            # Apply API constraints on top_k and ef
            requested_top_k = query.similarity_top_k if query.similarity_top_k is not None else 10
            top_k = min(requested_top_k, MAX_TOP_K_ALLOWED)
            if requested_top_k > MAX_TOP_K_ALLOWED:
                logger.warning(
                    f"Requested top_k ({requested_top_k}) exceeds maximum ({MAX_TOP_K_ALLOWED}). "
                    f"Capping at {MAX_TOP_K_ALLOWED}."
                )

            ef_capped = min(ef, MAX_EF_SEARCH_ALLOWED)
            if ef > MAX_EF_SEARCH_ALLOWED:
                logger.warning(
                    f"Requested ef ({ef}) exceeds maximum ({MAX_EF_SEARCH_ALLOWED}). "
                    f"Capping at {MAX_EF_SEARCH_ALLOWED}."
                )

            # Build and execute query
            query_kwargs = self._build_query_params(
                query_embedding, top_k, ef_capped, filter_for_api,
                sparse_indices_q, sparse_values_q,
                prefilter_cardinality_threshold, filter_boost_percentage,
            )

            try:
                logger.debug(f"Executing query: top_k={top_k}, ef={ef_capped}, filters={bool(filter_for_api)}")
                results = self._endee_index.query(**query_kwargs)
                logger.debug(f"Query returned {len(results)} results")
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            # Process results
            nodes, similarities, ids = self._process_query_results(results)
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Query failed with unexpected error: {e}")
            raise RuntimeError(f"Query execution failed: {e}") from e