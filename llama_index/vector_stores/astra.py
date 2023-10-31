"""Astra DB Vector store index.

An index based on a DB table with vector search capabilities,
powered by the astrapy library

"""

import logging

from typing import Any, Dict, Iterable, List, Optional, TypeVar, cast

from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

DEFAULT_INSERTION_BATCH_SIZE = 20

T = TypeVar("T")


def _batch_iterable(iterable: Iterable[T], batch_size: int) -> Iterable[Iterable[T]]:
    this_batch = []
    for entry in iterable:
        this_batch.append(entry)
        if len(this_batch) == batch_size:
            yield this_batch
            this_batch = []
    if this_batch:
        yield this_batch


class AstraDBVectorStore(VectorStore):
    """Astra DB Vector Store.

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
        insertion_batch_size[Optional[int]]: Batch size for insertions.
            Default is 20

    """

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        *,
        collection_name: str,
        token: str,
        api_endpoint: str,
        embedding_dimension: int,
        namespace: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        insertion_batch_size: int = DEFAULT_INSERTION_BATCH_SIZE,
    ) -> None:
        import_err_msg = "`astrapy` package not found, please run `pip install astrapy`"

        # Try to import astrapy for use
        try:
            from astrapy.db import AstraDBCollection, AstraDB
        except ImportError:
            raise ImportError(import_err_msg)
        
        # Set all the required class parameters
        self._embedding_dimension = embedding_dimension
        self._ttl_seconds = ttl_seconds
        self._insertion_batch_size = insertion_batch_size

        _logger.debug("Creating the Astra DB table")

        # Build the Astra DB object
        self.astra_db = AstraDB(
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace
        )

        # Create the new collection
        self.astra_db.create_collection(
            collection_name=collection_name,
            dimension=embedding_dimension
        )

        # Connect to the newly created collection
        self.astra_db_collection = AstraDBCollection(
            collection_name=collection_name,
            astra_db=self.astra_db
        )

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index.

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
                    "$vector": node.get_embedding()
                }
            )

        _logger.debug(f"Adding {len(nodes_list)} rows to table")
        
        # Concurrent batching of inserts:
        for insertion_batch in _batch_iterable(
            nodes_list, batch_size=self._insertion_batch_size
        ):
            futures = []
            for document in insertion_batch:
                self.astra_db_collection.insert_one(document)
            for future in futures:
                _ = future.result()

        return [n["_id"] for n in nodes_list]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        _logger.debug("Deleting a document from the Astra table")

        self.astra_db_collection.delete(
            id=ref_doc_id,
            **delete_kwargs
        )

    @property
    def client(self) -> Any:
        """Return the underlying Astra vector table object."""
        return self.astra_db_collection

    @staticmethod
    def _query_filters_to_dict(query_filters: MetadataFilters) -> Dict[str, Any]:
        if any(not isinstance(f, ExactMatchFilter) for f in query_filters.filters):
            raise NotImplementedError("Only `ExactMatchFilter` filters are supported")

        return {f.key: f.value for f in query_filters.filters}

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""

        # Get the query embedding
        query_embedding = cast(List[float], query.query_embedding)

        # Set the parameters accordingly
        sort = {"$vector": query_embedding}
        options = {"limit": query.similarity_top_k}
        projection = {"$vector": 1, "$similarity": 1}

        # Call the find method of the Astra API
        matches = self.astra_db_collection.find(
            sort=sort,
            options=options,
            projection=projection
        )

        # We have three lists to return
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        # Get every match
        for my_match in matches["data"]["documents"]:
            # Fetch the document data based on the id
            # TODO: the content of the node is not returned by default. Do we have to call the API again like this?
            document = self.astra_db_collection.find_one(filter={"_id": my_match["_id"]})["data"]["document"]

            # Grab the node information
            my_match["_node_content"] = "{}"

            node = metadata_dict_to_node(my_match)
            node.set_content(document["content"])

            # Append to the respective lists
            top_k_nodes.append(node)
            top_k_ids.append(my_match["_id"])
            top_k_scores.append(my_match["$similarity"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            similarities=top_k_scores,
            ids=top_k_ids,
        )
