"""Astra DB Vector store index.

An index based on a DB table with vector search capabilities,
powered by the astrapy library

"""
import logging
from typing import Any, List, Optional, cast

from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

MAX_INSERT_BATCH_SIZE = 20


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
    ) -> None:
        import_err_msg = "`astrapy` package not found, please run `pip install astrapy`"

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

        # Create and connect to the newly created collection
        self._astra_db_collection = self._astra_db.create_collection(
            collection_name=collection_name, dimension=embedding_dimension
        )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        # Get the query embedding
        query_embedding = cast(List[float], query.query_embedding)

        # Set the parameters accordingly
        sort = {"$vector": query_embedding}
        options = {"limit": query.similarity_top_k}
        projection = {"$vector": 1, "$similarity": 1, "content": 1}

        # Call the find method of the Astra API
        matches = self._astra_db_collection.find(
            sort=sort, options=options, projection=projection
        )["data"]["documents"]

        # We have three lists to return
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        # Get every match
        for my_match in matches:
            # Grab the node information
            my_match["_node_content"] = "{}"

            node = metadata_dict_to_node(my_match)
            node.set_content(my_match["content"])

            # Append to the respective lists
            top_k_nodes.append(node)
            top_k_ids.append(my_match["_id"])
            top_k_scores.append(my_match["$similarity"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            similarities=top_k_scores,
            ids=top_k_ids,
        )
