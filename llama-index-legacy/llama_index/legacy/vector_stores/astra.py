"""
Astra DB Vector store index.

An index based on a DB table with vector search capabilities,
powered by the astrapy library

"""

import json
import logging
from typing import Any, Dict, List, Optional, cast
from warnings import warn

from llama_index.legacy.bridge.pydantic import PrivateAttr
from llama_index.legacy.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.legacy.schema import BaseNode, MetadataMode
from llama_index.legacy.vector_stores.types import (
    BasePydanticVectorStore,
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

DEFAULT_MMR_PREFETCH_FACTOR = 4.0
MAX_INSERT_BATCH_SIZE = 20

NON_INDEXED_FIELDS = ["metadata._node_content", "content"]


class AstraDBVectorStore(BasePydanticVectorStore):
    """
    Astra DB Vector Store.

    An abstraction of a Astra table with
    vector-similarity-search. Documents, and their embeddings, are stored
    in an Astra table and a vector-capable index is used for searches.
    The table does not need to exist beforehand: if necessary it will
    be created behind the scenes.

    All Astra operations are done through the astrapy library.

    Args:
        collection_name (str): collection name to use. If not existing, it will be created.
        token (str): The Astra DB Application Token to use.
        api_endpoint (str): The Astra DB JSON API endpoint for your database.
        embedding_dimension (int): length of the embedding vectors in use.
        namespace (Optional[str]): The namespace to use. If not provided, 'default_keyspace'
        ttl_seconds (Optional[int]): expiration time for inserted entries.
            Default is no expiration.

    """

    stores_text: bool = True
    flat_metadata: bool = True

    _embedding_dimension: int = PrivateAttr()
    _ttl_seconds: Optional[int] = PrivateAttr()
    _astra_db: Any = PrivateAttr()
    _astra_db_collection: Any = PrivateAttr()

    def __init__(
        self,
        *,
        collection_name: str,
        token: str,
        api_endpoint: str,
        embedding_dimension: int,
        namespace: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        super().__init__()

        import_err_msg = (
            "`astrapy` package not found, please run `pip install --upgrade astrapy`"
        )

        # Try to import astrapy for use
        try:
            from astrapy.db import AstraDB
        except ImportError:
            raise ImportError(import_err_msg)

        # Set all the required class parameters
        self._embedding_dimension = embedding_dimension
        self._ttl_seconds = ttl_seconds

        _logger.debug("Creating the Astra DB table")

        # Build the Astra DB object
        self._astra_db = AstraDB(
            api_endpoint=api_endpoint, token=token, namespace=namespace
        )

        from astrapy.api import APIRequestError

        try:
            # Create and connect to the newly created collection
            self._astra_db_collection = self._astra_db.create_collection(
                collection_name=collection_name,
                dimension=embedding_dimension,
                options={"indexing": {"deny": NON_INDEXED_FIELDS}},
            )
        except APIRequestError as e:
            # possibly the collection is preexisting and has legacy
            # indexing settings: verify
            get_coll_response = self._astra_db.get_collections(
                options={"explain": True}
            )
            collections = (get_coll_response["status"] or {}).get("collections") or []
            preexisting = [
                collection
                for collection in collections
                if collection["name"] == collection_name
            ]
            if preexisting:
                pre_collection = preexisting[0]
                # if it has no "indexing", it is a legacy collection;
                # otherwise it's unexpected warn and proceed at user's risk
                pre_col_options = pre_collection.get("options") or {}
                if "indexing" not in pre_col_options:
                    warn(
                        (
                            f"Collection '{collection_name}' is detected as legacy"
                            " and has indexing turned on for all fields. This"
                            " implies stricter limitations on the amount of text"
                            " each entry can store. Consider reindexing anew on a"
                            " fresh collection to be able to store longer texts."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._astra_db_collection = self._astra_db.collection(
                        collection_name=collection_name,
                    )
                else:
                    options_json = json.dumps(pre_col_options["indexing"])
                    warn(
                        (
                            f"Collection '{collection_name}' has unexpected 'indexing'"
                            f" settings (options.indexing = {options_json})."
                            " This can result in odd behaviour when running "
                            " metadata filtering and/or unwarranted limitations"
                            " on storing long texts. Consider reindexing anew on a"
                            " fresh collection."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._astra_db_collection = self._astra_db.collection(
                        collection_name=collection_name,
                    )
            else:
                # other exception
                raise

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of node with embeddings

        """
        # Initialize list of objects to track
        nodes_list = []

        # Process each node individually
        for node in nodes:
            # Get the metadata
            metadata = node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            )

            # One dictionary of node data per node
            nodes_list.append(
                {
                    "_id": node.node_id,
                    "content": node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": metadata,
                    "$vector": node.get_embedding(),
                }
            )

        # Log the number of rows being added
        _logger.debug(f"Adding {len(nodes_list)} rows to table")

        # Initialize an empty list to hold the batches
        batched_list = []

        # Iterate over the node_list in steps of MAX_INSERT_BATCH_SIZE
        for i in range(0, len(nodes_list), MAX_INSERT_BATCH_SIZE):
            # Append a slice of node_list to the batched_list
            batched_list.append(nodes_list[i : i + MAX_INSERT_BATCH_SIZE])

        # Perform the bulk insert
        for i, batch in enumerate(batched_list):
            _logger.debug(f"Processing batch #{i + 1} of size {len(batch)}")

            # Go to astrapy to perform the bulk insert
            self._astra_db_collection.insert_many(batch)

        # Return the list of ids
        return [str(n["_id"]) for n in nodes_list]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        _logger.debug("Deleting a document from the Astra table")

        self._astra_db_collection.delete(id=ref_doc_id, **delete_kwargs)

    @property
    def client(self) -> Any:
        """Return the underlying Astra vector table object."""
        return self._astra_db_collection

    @staticmethod
    def _query_filters_to_dict(query_filters: MetadataFilters) -> Dict[str, Any]:
        # Allow only legacy ExactMatchFilter and MetadataFilter with FilterOperator.EQ
        if not all(
            (
                isinstance(f, ExactMatchFilter)
                or (isinstance(f, MetadataFilter) and f.operator == FilterOperator.EQ)
            )
            for f in query_filters.filters
        ):
            raise NotImplementedError(
                "Only filters with operator=FilterOperator.EQ are supported"
            )
        return {f"metadata.{f.key}": f.value for f in query_filters.filters}

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        # Get the currently available query modes
        _available_query_modes = [
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.MMR,
        ]

        # Reject query if not available
        if query.mode not in _available_query_modes:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        # Get the query embedding
        query_embedding = cast(List[float], query.query_embedding)

        # Process the metadata filters as needed
        if query.filters is not None:
            query_metadata = self._query_filters_to_dict(query.filters)
        else:
            query_metadata = {}

        # Get the scores depending on the query mode
        if query.mode == VectorStoreQueryMode.DEFAULT:
            # Call the vector_find method of AstraPy
            matches = self._astra_db_collection.vector_find(
                vector=query_embedding,
                limit=query.similarity_top_k,
                filter=query_metadata,
            )

            # Get the scores associated with each
            top_k_scores = [match["$similarity"] for match in matches]
        elif query.mode == VectorStoreQueryMode.MMR:
            # Querying a larger number of vectors and then doing MMR on them.
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
            # Get the most we can possibly need to fetch
            prefetch_k = max(prefetch_k0, query.similarity_top_k)

            # Call AstraPy to fetch them
            prefetch_matches = self._astra_db_collection.vector_find(
                vector=query_embedding,
                limit=prefetch_k,
                filter=query_metadata,
            )

            # Get the MMR threshold
            mmr_threshold = query.mmr_threshold or kwargs.get("mmr_threshold")

            # If we have found documents, we can proceed
            if prefetch_matches:
                zipped_indices, zipped_embeddings = zip(
                    *enumerate(match["$vector"] for match in prefetch_matches)
                )
                pf_match_indices, pf_match_embeddings = list(zipped_indices), list(
                    zipped_embeddings
                )
            else:
                pf_match_indices, pf_match_embeddings = [], []

            # Call the Llama utility function to get the top  k
            mmr_similarities, mmr_indices = get_top_k_mmr_embeddings(
                query_embedding,
                pf_match_embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=pf_match_indices,
                mmr_threshold=mmr_threshold,
            )

            # Finally, build the final results based on the mmr values
            matches = [prefetch_matches[mmr_index] for mmr_index in mmr_indices]
            top_k_scores = mmr_similarities

        # We have three lists to return
        top_k_nodes = []
        top_k_ids = []

        # Get every match
        for match in matches:
            # Check whether we have a llama-generated node content field
            if "_node_content" not in match["metadata"]:
                match["metadata"]["_node_content"] = json.dumps(match)

            # Create a new node object from the node metadata
            node = metadata_dict_to_node(match["metadata"], text=match["content"])

            # Append to the respective lists
            top_k_nodes.append(node)
            top_k_ids.append(match["_id"])

        # return our final result
        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            similarities=top_k_scores,
            ids=top_k_ids,
        )
