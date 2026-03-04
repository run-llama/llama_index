"""Chroma vector store."""

import logging
import math
from typing import Any, Dict, Generator, List, Optional, Union, cast

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import truncate_text
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

# MMR constants
DEFAULT_MMR_PREFETCH_FACTOR = 4.0


def _transform_chroma_filter_condition(condition: str) -> str:
    """Translate standard metadata filter op to Chroma specific spec."""
    if condition == "and":
        return "$and"
    elif condition == "or":
        return "$or"
    else:
        raise ValueError(f"Filter condition {condition} not supported")


def _transform_chroma_filter_operator(operator: str) -> str:
    """Translate standard metadata filter operator to Chroma specific spec."""
    if operator == "!=":
        return "$ne"
    elif operator == "==":
        return "$eq"
    elif operator == ">":
        return "$gt"
    elif operator == "<":
        return "$lt"
    elif operator == ">=":
        return "$gte"
    elif operator == "<=":
        return "$lte"
    elif operator == "in":
        return "$in"
    elif operator == "nin":
        return "$nin"
    else:
        raise ValueError(f"Filter operator {operator} not supported")


def _to_chroma_filter(
    standard_filters: MetadataFilters,
) -> dict:
    """Translate standard metadata filters to Chroma specific spec."""
    filters = {}
    filters_list = []
    condition = standard_filters.condition or "and"
    condition = _transform_chroma_filter_condition(condition)
    if standard_filters.filters:
        for filter in standard_filters.filters:
            if isinstance(filter, MetadataFilters):
                filters_list.append(_to_chroma_filter(filter))
            elif filter.operator:
                filters_list.append(
                    {
                        filter.key: {
                            _transform_chroma_filter_operator(
                                filter.operator
                            ): filter.value
                        }
                    }
                )
            else:
                filters_list.append({filter.key: filter.value})

    if len(filters_list) == 1:
        # If there is only one filter, return it directly
        return filters_list[0]
    elif len(filters_list) > 1:
        filters[condition] = filters_list
    return filters


import_err_msg = "`chromadb` package not found, please run `pip install chromadb`"

MAX_CHUNK_SIZE = 41665  # One less than the max chunk size for ChromaDB


def chunk_list(
    lst: List[BaseNode], max_chunk_size: int
) -> Generator[List[BaseNode], None, None]:
    """
    Yield successive max_chunk_size-sized chunks from lst.

    Args:
        lst (List[BaseNode]): list of nodes with embeddings
        max_chunk_size (int): max chunk size

    Yields:
        Generator[List[BaseNode], None, None]: list of nodes with embeddings

    """
    for i in range(0, len(lst), max_chunk_size):
        yield lst[i : i + max_chunk_size]


class ChromaVectorStore(BasePydanticVectorStore):
    """
    Chroma vector store.

    In this vector store, embeddings are stored within a ChromaDB collection.

    During query time, the index uses ChromaDB to query for the top
    k most similar nodes.

    Supports MMR (Maximum Marginal Relevance) search mode for improved diversity
    in search results.

    Args:
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    Examples:
        `uv add llama-index-vector-stores-chroma`

        ```python
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Create a Chroma client and collection
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("example_collection")

        # Set up the ChromaVectorStore and StorageContext
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Use MMR mode with threshold
        query_engine = index.as_query_engine(
            vector_store_query_mode="mmr",
            vector_store_kwargs={"mmr_threshold": 0.5}
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    collection_name: Optional[str]
    host: Optional[str]
    port: Optional[Union[str, int]]
    ssl: bool
    headers: Optional[Dict[str, str]]
    persist_dir: Optional[str]
    collection_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _collection: Collection = PrivateAttr()

    def __init__(
        self,
        chroma_collection: Optional[Any] = None,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        persist_dir: Optional[str] = None,
        collection_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        collection_kwargs = collection_kwargs or {}

        super().__init__(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            collection_name=collection_name,
            persist_dir=persist_dir,
            collection_kwargs=collection_kwargs or {},
        )
        if chroma_collection is None:
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            self._collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        else:
            self._collection = cast(Collection, chroma_collection)

    @classmethod
    def from_collection(cls, collection: Any) -> "ChromaVectorStore":
        try:
            from chromadb import Collection
        except ImportError:
            raise ImportError(import_err_msg)

        if not isinstance(collection, Collection):
            raise Exception("argument is not chromadb collection instance")

        return cls(chroma_collection=collection)

    @classmethod
    def from_params(
        cls,
        collection_name: str,
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        persist_dir: Optional[str] = None,
        collection_kwargs: dict = {},
        **kwargs: Any,
    ) -> "ChromaVectorStore":
        if persist_dir:
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        elif host and port:
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
            collection = client.get_or_create_collection(
                name=collection_name, **collection_kwargs
            )
        else:
            raise ValueError(
                "Either `persist_dir` or (`host`,`port`) must be specified"
            )
        return cls(
            chroma_collection=collection,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            persist_dir=persist_dir,
            collection_kwargs=collection_kwargs,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ChromaVectorStore"

    def get_nodes(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[List[MetadataFilters]] = None,
    ) -> List[BaseNode]:
        """
        Get nodes from index.

        Args:
            node_ids (List[str]): list of node ids
            filters (List[MetadataFilters]): list of metadata filters

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        node_ids = node_ids or None

        if filters:
            where = _to_chroma_filter(filters)
        else:
            where = None

        result = self._get(None, where=where, ids=node_ids)

        return result.nodes

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        max_chunk_size = MAX_CHUNK_SIZE
        node_chunks = chunk_list(nodes, max_chunk_size)

        all_ids = []
        for node_chunk in node_chunks:
            embeddings = []
            metadatas = []
            ids = []
            documents = []
            for node in node_chunk:
                embeddings.append(node.get_embedding())
                metadata_dict = node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
                for key in metadata_dict:
                    if metadata_dict[key] is None:
                        metadata_dict[key] = ""
                metadatas.append(metadata_dict)
                ids.append(node.node_id)
                documents.append(node.get_content(metadata_mode=MetadataMode.NONE))

            self._collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            all_ids.extend(ids)

        return all_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._collection.delete(where={"document_id": ref_doc_id})

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[List[MetadataFilters]] = None,
    ) -> None:
        """
        Delete nodes from index.

        Args:
            node_ids (List[str]): list of node ids
            filters (List[MetadataFilters]): list of metadata filters

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        node_ids = node_ids or []

        if filters:
            where = _to_chroma_filter(filters)
            self._collection.delete(ids=node_ids, where=where)

        else:
            self._collection.delete(ids=node_ids)

    def clear(self) -> None:
        """Clear the collection."""
        ids = self._collection.get()["ids"]
        self._collection.delete(ids=ids)

    @property
    def client(self) -> Any:
        """Return client."""
        return self._collection

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): Query object containing:
                - query_embedding (List[float]): query embedding
                - similarity_top_k (int): top k most similar nodes
                - filters (Optional[MetadataFilters]): metadata filters to apply
                - mode (VectorStoreQueryMode): query mode (default or MMR)
            **kwargs: Additional keyword arguments passed to ChromaDB query method.
                For MMR mode, supports:
                - mmr_threshold (Optional[float]): MMR threshold between 0 and 1
                - mmr_prefetch_factor (Optional[float]): Factor to multiply similarity_top_k
                for prefetching candidates (default: 4.0)
                - mmr_prefetch_k (Optional[int]): Explicit number of candidates to prefetch
                (cannot be used with mmr_prefetch_factor)
                For ChromaDB-specific parameters:
                - where (dict): ChromaDB where clause (use query.filters instead for standard filtering)
                - include (List[str]): ChromaDB include parameter
                - where_document (dict): ChromaDB where_document parameter

        Returns:
            VectorStoreQueryResult: Query result containing matched nodes, similarities, and IDs.

        Raises:
            ValueError: If MMR parameters are invalid or if both query.filters and
                    where kwargs are specified.

        """
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify metadata filters via both query and kwargs. "
                    "Use kwargs only for chroma specific items that are "
                    "not supported via the generic query interface."
                )
            where = _to_chroma_filter(query.filters)
        else:
            where = kwargs.pop("where", None)

        if not query.query_embedding:
            return self._get(limit=query.similarity_top_k, where=where, **kwargs)

        # Handle MMR mode
        if query.mode == VectorStoreQueryMode.MMR:
            return self._mmr_search(query, where, **kwargs)

        return self._query(
            query_embeddings=query.query_embedding,
            n_results=query.similarity_top_k,
            where=where,
            **kwargs,
        )

    def _query(
        self, query_embeddings: List["float"], n_results: int, where: dict, **kwargs
    ) -> VectorStoreQueryResult:
        if where:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                **kwargs,
            )
        else:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                **kwargs,
            )

        logger.debug(f"> Top {len(results['documents'][0])} nodes:")
        nodes = []
        similarities = []
        ids = []
        for node_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            try:
                node = metadata_dict_to_node(metadata, text=text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=text or "",
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)

            similarity_score = math.exp(-distance)
            similarities.append(similarity_score)

            logger.debug(
                f"> [Node {node_id}] [Similarity score: {similarity_score}] "
                f"{truncate_text(str(text), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _mmr_search(
        self, query: VectorStoreQuery, where: dict, **kwargs
    ) -> VectorStoreQueryResult:
        """
        Perform MMR search using ChromaDB.

        Args:
            query: VectorStoreQuery object containing the query parameters
            where: ChromaDB filter conditions
            **kwargs: Additional keyword arguments including mmr_threshold

        Returns:
            VectorStoreQueryResult: Query result with MMR-applied nodes

        """
        # Extract MMR parameters
        mmr_threshold = kwargs.get("mmr_threshold")

        # Validate MMR parameters
        if mmr_threshold is not None and (
            not isinstance(mmr_threshold, (int, float))
            or mmr_threshold < 0
            or mmr_threshold > 1
        ):
            raise ValueError("mmr_threshold must be a float between 0 and 1")

        # Validate prefetch parameters (check before popping)
        raw_prefetch_factor = kwargs.get("mmr_prefetch_factor")
        raw_prefetch_k = kwargs.get("mmr_prefetch_k")
        if raw_prefetch_factor is not None and raw_prefetch_k is not None:
            raise ValueError(
                "'mmr_prefetch_factor' and 'mmr_prefetch_k' "
                "cannot coexist in a call to query()"
            )

        # Strip MMR-only kwargs so they aren't forwarded to Chroma
        mmr_threshold = kwargs.pop("mmr_threshold", None)
        prefetch_k_override = kwargs.pop("mmr_prefetch_k", None)
        prefetch_factor = kwargs.pop("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)

        # Calculate prefetch size (get more candidates than needed for MMR)
        if prefetch_k_override is not None:
            prefetch_k = int(prefetch_k_override)
        else:
            prefetch_k = int(query.similarity_top_k * prefetch_factor)

        # Ensure prefetch_k is at least as large as similarity_top_k
        prefetch_k = max(prefetch_k, query.similarity_top_k)

        logger.debug(
            f"MMR search: prefetching {prefetch_k} candidates for {query.similarity_top_k} final results"
        )

        # Query ChromaDB for more candidates than needed (kwargs now safe)
        if where:
            prefetch_results = self._collection.query(
                query_embeddings=query.query_embedding,
                n_results=prefetch_k,
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"],
                **kwargs,
            )
        else:
            prefetch_results = self._collection.query(
                query_embeddings=query.query_embedding,
                n_results=prefetch_k,
                include=["embeddings", "documents", "metadatas", "distances"],
                **kwargs,
            )

        # Extract embeddings and metadata for MMR processing
        prefetch_embeddings = []
        prefetch_ids = []
        prefetch_metadata = []
        prefetch_documents = []
        prefetch_distances = []

        # Process prefetch results
        for i in range(len(prefetch_results["ids"][0])):
            node_id = prefetch_results["ids"][0][i]
            text = prefetch_results["documents"][0][i]
            metadata = prefetch_results["metadatas"][0][i]
            distance = prefetch_results["distances"][0][i]

            # Get the actual embedding from ChromaDB results
            if "embeddings" in prefetch_results and prefetch_results["embeddings"]:
                embedding = prefetch_results["embeddings"][0][i]
            else:
                # Fallback: if embeddings not available, we'll use distance-based approach
                embedding = None

            # Store for MMR processing
            prefetch_embeddings.append(embedding)
            prefetch_ids.append(node_id)
            prefetch_metadata.append(metadata)
            prefetch_documents.append(text)
            prefetch_distances.append(distance)

        if not prefetch_embeddings:
            logger.warning("No results found during MMR prefetch")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Check if we have valid embeddings for MMR
        valid_embeddings = [emb for emb in prefetch_embeddings if emb is not None]

        if len(valid_embeddings) < query.similarity_top_k:
            logger.warning(
                f"Not enough valid embeddings for MMR: {len(valid_embeddings)} < {query.similarity_top_k}"
            )
            # Fallback to regular similarity search
            return self._query(
                query_embeddings=query.query_embedding,
                n_results=query.similarity_top_k,
                where=where,
                **kwargs,
            )

        # Apply MMR algorithm using the core utility function
        mmr_similarities, mmr_indices = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=valid_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=list(range(len(valid_embeddings))),
            mmr_threshold=mmr_threshold,
        )

        # Build final results based on MMR selection
        final_nodes = []
        final_similarities = []
        final_ids = []

        # Create a mapping from valid embedding indices to original prefetch indices
        valid_indices = [
            i for i, emb in enumerate(prefetch_embeddings) if emb is not None
        ]

        for mmr_index in mmr_indices:
            if mmr_index < len(valid_indices):
                original_index = valid_indices[mmr_index]
                if original_index < len(prefetch_ids):
                    node_id = prefetch_ids[original_index]
                    text = prefetch_documents[original_index]
                    metadata = prefetch_metadata[original_index]
                    distance = prefetch_distances[original_index]

                    # Create node (reusing logic from _query method)
                    try:
                        node = metadata_dict_to_node(metadata, text=text)
                    except Exception:
                        # NOTE: deprecated legacy logic for backward compatibility
                        metadata, node_info, relationships = (
                            legacy_metadata_dict_to_node(metadata)
                        )

                        node = TextNode(
                            text=text or "",
                            id_=node_id,
                            metadata=metadata,
                            start_char_idx=node_info.get("start", None),
                            end_char_idx=node_info.get("end", None),
                            relationships=relationships,
                        )

                    final_nodes.append(node)
                    final_similarities.append(math.exp(-distance))
                    final_ids.append(node_id)

        logger.debug(
            f"MMR search completed: {len(final_nodes)} results selected from {len(prefetch_embeddings)} candidates"
        )

        return VectorStoreQueryResult(
            nodes=final_nodes, similarities=final_similarities, ids=final_ids
        )

    def _get(
        self, limit: Optional[int], where: dict, **kwargs
    ) -> VectorStoreQueryResult:
        if where:
            results = self._collection.get(
                limit=limit,
                where=where,
                **kwargs,
            )
        else:
            results = self._collection.get(
                limit=limit,
                **kwargs,
            )

        logger.debug(f"> Top {len(results['documents'])} nodes:")
        nodes = []
        ids = []

        if not results["ids"]:
            results["ids"] = [[]]

        for node_id, text, metadata in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            try:
                node = metadata_dict_to_node(metadata, text=text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=text or "",
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)

            logger.debug(
                f"> [Node {node_id}] [Similarity score: N/A - using get()] "
                f"{truncate_text(str(text), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, ids=ids)
