"""
Astra DB Vector Store index.

An index based on a DB collection with vector search capabilities,
powered by the AstraPy library

"""

import json
import logging
from typing import Any, Dict, List, Optional, cast
from concurrent.futures import ThreadPoolExecutor
from warnings import warn

from astrapy import DataAPIClient
from astrapy.exceptions import InsertManyException
from astrapy.results import UpdateResult

import llama_index.core
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

DEFAULT_MMR_PREFETCH_FACTOR = 4.0

REPLACE_DOCUMENTS_MAX_THREADS = 12

NON_INDEXED_FIELDS = ["metadata._node_content", "content"]


class AstraDBVectorStore(BasePydanticVectorStore):
    """
    Astra DB Vector Store.

    An abstraction of a Astra DB collection with
    vector-similarity-search. Documents, and their embeddings, are stored
    in an Astra DB collection equipped with a vector index.
    The collection, if necessary, is created when the vector store is initialized.

    All Astra operations are done through the AstraPy library.

    Visit https://astra.datastax.com/signup to create an account and get started.

    Args:
        collection_name (str): collection name to use. If not existing, it will be created.
        token (str): The Astra DB Application Token to use.
        api_endpoint (str): The Astra DB JSON API endpoint for your database.
        embedding_dimension (int): length of the embedding vectors in use.
        keyspace (Optional[str]): The keyspace to use. If not provided, 'default_keyspace'
        namespace (Optional[str]): [DEPRECATED] The keyspace to use. If not provided, 'default_keyspace'

    Examples:
        `pip install llama-index-vector-stores-astra`

        ```python
        from llama_index.vector_stores.astra import AstraDBVectorStore

        # Create the Astra DB Vector Store object
        astra_db_store = AstraDBVectorStore(
            collection_name="astra_v_store",
            token=token,
            api_endpoint=api_endpoint,
            embedding_dimension=1536,
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    _embedding_dimension: int = PrivateAttr()
    _database: Any = PrivateAttr()
    _collection: Any = PrivateAttr()

    def __init__(
        self,
        *,
        collection_name: str,
        token: str,
        api_endpoint: str,
        embedding_dimension: int,
        keyspace: Optional[str] = None,
        namespace: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Set all the required class parameters
        self._embedding_dimension = embedding_dimension

        if ttl_seconds is not None:
            warn(
                (
                    "Parameter `ttl_seconds` is not supported for "
                    "`AstraDBVectorStore` and will be ignored."
                ),
                UserWarning,
                stacklevel=2,
            )

        _logger.debug("Creating the Astra DB client and database instances")

        # Choose the keyspace param
        keyspace_param = keyspace or namespace

        # Build the Database object
        self._database = DataAPIClient(
            caller_name=getattr(llama_index, "__name__", "llama_index"),
            caller_version=getattr(llama_index.core, "__version__", None),
        ).get_database(
            api_endpoint,
            token=token,
            keyspace=keyspace_param,
        )

        from astrapy.exceptions import DataAPIException

        collection_indexing = {"deny": NON_INDEXED_FIELDS}

        try:
            _logger.debug("Creating the Astra DB collection")
            # Create and connect to the newly created collection
            self._collection = self._database.create_collection(
                name=collection_name,
                dimension=embedding_dimension,
                indexing=collection_indexing,
                check_exists=False,
            )
        except DataAPIException as e:
            # possibly the collection is preexisting and has legacy
            # indexing settings: verify
            preexisting = [
                coll_descriptor
                for coll_descriptor in self._database.list_collections()
                if coll_descriptor.name == collection_name
            ]
            if preexisting:
                # if it has no "indexing", it is a legacy collection;
                # otherwise it's unexpected: warn and proceed at user's risk
                pre_col_idx_opts = preexisting[0].options.indexing or {}
                if not pre_col_idx_opts:
                    warn(
                        (
                            f"Collection '{collection_name}' is detected as "
                            "having indexing turned on for all fields "
                            "(either created manually or by older versions "
                            "of this plugin). This implies stricter "
                            "limitations on the amount of text"
                            " each entry can store. Consider indexing anew on a"
                            " fresh collection to be able to store longer texts."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._collection = self._database.get_collection(
                        collection_name,
                    )
                else:
                    # check if the indexing options match entirely
                    if pre_col_idx_opts == collection_indexing:
                        raise
                    else:
                        options_json = json.dumps(pre_col_idx_opts)
                        warn(
                            (
                                f"Collection '{collection_name}' has unexpected 'indexing'"
                                f" settings (options.indexing = {options_json})."
                                " This can result in odd behaviour when running "
                                " metadata filtering and/or unwarranted limitations"
                                " on storing long texts. Consider indexing anew on a"
                                " fresh collection."
                            ),
                            UserWarning,
                            stacklevel=2,
                        )
                        self._collection = self._database.get_collection(
                            collection_name,
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
        # Initialize list of documents to insert
        documents_to_insert: List[Dict[str, Any]] = []

        # Process each node individually
        for node in nodes:
            # Get the metadata
            metadata = node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            )

            # One dictionary of node data per node
            documents_to_insert.append(
                {
                    "_id": node.node_id,
                    "content": node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": metadata,
                    "$vector": node.get_embedding(),
                }
            )

        # Log the number of documents being added
        _logger.debug(f"Adding {len(documents_to_insert)} documents to the collection")

        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: List[int]
        try:
            self._collection.insert_many(
                documents_to_insert,
                ordered=False,
            )
            ids_to_replace = []
        except InsertManyException as err:
            inserted_ids_set = set(err.partial_result.inserted_ids)
            ids_to_replace = [
                document["_id"]
                for document in documents_to_insert
                if document["_id"] not in inserted_ids_set
            ]
            _logger.debug(
                f"Detected {len(ids_to_replace)} non-inserted documents, trying replace_one"
            )

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if document["_id"] in ids_to_replace
            ]

            with ThreadPoolExecutor(
                max_workers=REPLACE_DOCUMENTS_MAX_THREADS
            ) as executor:

                def _replace_document(document: Dict[str, Any]) -> UpdateResult:
                    return self._collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    )

                replace_results = executor.map(
                    _replace_document,
                    documents_to_replace,
                )

            replaced_count = sum(r_res.update_info["n"] for r_res in replace_results)
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                raise ValueError(
                    "AstraDBVectorStore.add could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )

        # Return the list of ids
        return [str(n["_id"]) for n in documents_to_insert]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        _logger.debug("Deleting a document from the Astra DB collection")

        if delete_kwargs:
            args_desc = ", ".join(
                f"'{kwarg}'" for kwarg in sorted(delete_kwargs.keys())
            )
            warn(
                (
                    "AstraDBVectorStore.delete call got unsupported "
                    f"named argument(s): {args_desc}."
                ),
                UserWarning,
                stacklevel=2,
            )

        self._collection.delete_one({"_id": ref_doc_id})

    @property
    def client(self) -> Any:
        """Return the underlying Astra DB `astrapy.Collection` object."""
        return self._collection

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
        # nested filters, i.e. f being of type MetadataFilters, is excluded above:
        return {f"metadata.{f.key}": f.value for f in query_filters.filters}  # type: ignore[union-attr]

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

        matches: List[Dict[str, Any]]

        # Get the scores depending on the query mode
        if query.mode == VectorStoreQueryMode.DEFAULT:
            # Call the vector_find method of AstraPy
            matches = list(
                self._collection.find(
                    filter=query_metadata,
                    projection={"*": True},
                    limit=query.similarity_top_k,
                    sort={"$vector": query_embedding},
                    include_similarity=True,
                )
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

            # Call AstraPy to fetch them (similarity from DB not needed here)
            prefetch_matches = list(
                self._collection.find(
                    filter=query_metadata,
                    projection={"*": True},
                    limit=prefetch_k,
                    sort={"$vector": query_embedding},
                )
            )

            # Get the MMR threshold
            mmr_threshold = query.mmr_threshold or kwargs.get("mmr_threshold")

            # If we have found documents, we can proceed
            if prefetch_matches:
                zipped_indices, zipped_embeddings = zip(
                    *enumerate(match["$vector"] for match in prefetch_matches)
                )
                pf_match_indices, pf_match_embeddings = (
                    list(zipped_indices),
                    list(zipped_embeddings),
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
