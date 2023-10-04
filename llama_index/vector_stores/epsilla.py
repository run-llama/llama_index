"""Epsilla vector store."""
import logging
from typing import Any, List, Optional

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class EpsillaVectorStore(VectorStore):
    """The Epsilla Vector Store.

    In this vector store we store the text, its embedding and
    a few pieces of its metadata in a Epsilla collection. This implemnetation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection does not
    exist or if `overwrite` is set to True.

    As a prerequisite, you need to install ``pyepsilla`` package
    and have a running Epsilla vector database (for example, through our docker image)
    See the following documentation for how to run an Epsilla vector database:
    https://epsilla-inc.gitbook.io/epsilladb/quick-start

    Args:
        client (Any): Epsilla client to connect to.
        collection_name (Optional[str]): Which collection to use.
                    Defaults to "llama_collection".
        db_path (Optional[str]): The path where the database will be persisted.
                    Defaults to "/tmp/langchain-epsilla".
        db_name (Optional[str]): Give a name to the loaded database.
                    Defaults to "langchain_store".
        dimension (Optional[int]): The dimension of the embeddings. If not provided,
                    collection creation will be done on first insert. Defaults to None.
        overwrite (Optional[bool]): Whether to overwrite existing collection with same
                    name. Defaults to False.

    Returns:
        EpsillaVectorStore: Vectorstore that supports add, delete, and query.
    """

    stores_text = True
    flat_metadata: bool = False

    def __init__(
        self,
        client: Any,
        collection_name: str = "llama_collection",
        db_path: Optional[str] = DEFAULT_PERSIST_DIR,  # sub folder
        db_name: Optional[str] = "llama_db",
        dimension: Optional[int] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            from pyepsilla import vectordb
        except ImportError as e:
            raise ImportError(
                "Could not import pyepsilla python package. "
                "Please install pyepsilla package with `pip/pip3 install pyepsilla`."
            ) from e

        if not isinstance(client, vectordb.Client):
            raise TypeError(
                f"client should be an instance of pyepsilla.vectordb.Client, "
                f"got {type(client)}"
            )

        self._client: vectordb.Client = client
        self._collection_name = collection_name
        self._client.load_db(db_name, db_path)
        self._client.use_db(db_name)
        self._collection_created = False

        status_code, response = self._client.list_tables()
        if status_code != 200:
            self._handle_error(msg=response["message"])
        table_list = response["result"]

        if self._collection_name in table_list and overwrite is False:
            self._collection_created = True

        if self._collection_name in table_list and overwrite is True:
            status_code, response = self._client.drop_table(
                table_name=self._collection_name
            )
            if status_code != 200:
                self._handle_error(msg=response["message"])
            logger.debug(
                f"Successfully removed old collection: {self._collection_name}"
            )
            if dimension is not None:
                self._create_collection(dimension)

        if self._collection_name not in table_list and dimension is not None:
            self._create_collection(dimension)

    def client(self) -> Any:
        """Return the Epsilla client."""
        return self._client

    def _handle_error(self, msg: str) -> None:
        """Handle error."""
        logger.error(f"Failed to get records: {msg}")
        raise Exception(f"Error: {msg}.")

    def _create_collection(self, dimension: int) -> None:
        """
        Create collection.

        Args:
            dimension (int): The dimension of the embeddings.
        """
        fields: List[dict] = [
            {"name": "id", "dataType": "STRING", "primaryKey": True},
            {"name": DEFAULT_DOC_ID_KEY, "dataType": "STRING"},
            {"name": DEFAULT_TEXT_KEY, "dataType": "STRING"},
            {
                "name": DEFAULT_EMBEDDING_KEY,
                "dataType": "VECTOR_FLOAT",
                "dimensions": dimension,
            },
            {"name": "metadata", "dataType": "JSON"},
        ]
        status_code, response = self._client.create_table(
            table_name=self._collection_name, table_fields=fields
        )
        if status_code != 200:
            self._handle_error(msg=response["message"])
        self._collection_created = True
        logger.debug(f"Successfully created collection: {self._collection_name}")

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """
        Add nodes to Epsilla vector store.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            List[str]: List of ids inserted.
        """
        # If the collection doesn't exist yet, create the collection
        if not self._collection_created and len(nodes) > 0:
            dimension = len(nodes[0].get_embedding())
            self._create_collection(dimension)

        elif len(nodes) == 0:
            return []

        ids = []
        records = []
        for node in nodes:
            ids.append(node.node_id)
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata_dict = node_to_metadata_dict(node, remove_text=True)
            metadata = metadata_dict["_node_content"]
            record = {
                "id": node.node_id,
                DEFAULT_DOC_ID_KEY: node.ref_doc_id,
                DEFAULT_TEXT_KEY: text,
                DEFAULT_EMBEDDING_KEY: node.get_embedding(),
                "metadata": metadata,
            }
            records.append(record)

        status_code, response = self._client.insert(
            table_name=self._collection_name, records=records
        )
        if status_code != 200:
            self._handle_error(msg=response["message"])

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
        """
        raise NotImplementedError("Delete with filtering will be coming soon.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query.

        Returns:
            Vector store query result.
        """
        if not self._collection_created:
            raise ValueError("Please initialize a collection first.")

        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Epsilla does not support {query.mode} yet.")

        if query.filters is not None:
            raise NotImplementedError("Epsilla does not support Metadata filters yet.")

        if query.doc_ids is not None and len(query.doc_ids) > 0:
            raise NotImplementedError("Epsilla does not support filters yet.")

        status_code, response = self._client.query(
            table_name=self._collection_name,
            query_field=DEFAULT_EMBEDDING_KEY,
            query_vector=query.query_embedding,
            limit=query.similarity_top_k,
            with_distance=True,
        )
        if status_code != 200:
            self._handle_error(msg=response["message"])

        results = response["result"]
        logger.debug(
            f"Successfully searched embedding in collection: {self._collection_name}"
            f" Num Results: {len(results)}"
        )

        nodes = []
        similarities = []
        ids = []
        for res in results:
            try:
                node = metadata_dict_to_node({"_node_content": res["metadata"]})
                node.text = res[DEFAULT_TEXT_KEY]
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    res["metadata"]
                )
                node = TextNode(
                    id=res["id"],
                    text=res[DEFAULT_TEXT_KEY],
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            nodes.append(node)
            similarities.append(res["@distance"])
            ids.append(res["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
