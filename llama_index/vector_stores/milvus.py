"""Milvus vector store index.

An index that is built within Milvus.

"""
import logging
from typing import Any, List, Optional

from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

MILVUS_ID_FIELD = "id"


def _to_milvus_filter(standard_filters: MetadataFilters) -> List[str]:
    """Translate standard metadata filters to Milvus specific spec."""
    filters = []
    for filter in standard_filters.filters:
        if isinstance(filter.value, str):
            filters.append(str(filter.key) + " == " + '"' + str(filter.value) + '"')
        else:
            filters.append(str(filter.key) + " == " + str(filter.value))
    return filters


class MilvusVectorStore(VectorStore):
    """The Milvus Vector Store.

    In this vector store we store the text, its embedding and
    a its metadata in a Milvus collection. This implementation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesnt
    exist or if `overwrite` is set to True.

    Args:
        uri (str, optional): The URI to connect to, comes in the form of
            "http://address:port".
        token (str, optional): The token for log in. Empty if not using rbac, if
            using rbac it will most likely be "username:password".
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        dim (int, optional): The dimension of the embedding vectors for the collection.
            Required if creating a new colletion.
        embedding_field (str, optional): The name of the embedding field for the
            collection, defaults to DEFAULT_EMBEDDING_KEY.
        doc_id_field (str, optional): The name of the doc_id field for the collection,
            defaults to DEFAULT_DOC_ID_KEY.
        similarity_metric (str, optional): The similarity metric to use,
            currently supports IP and L2.
        consistency_level (str, optional): Which consistency level to use for a newly
            created collection. Defaults to "Session".
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.

    Raises:
        ImportError: Unable to import `pymilvus`.
        MilvusException: Error communicating with Milvus, more can be found in logging
            under Debug.

    Returns:
        MilvusVectorstore: Vectorstore that supports add, delete, and query.
    """

    stores_text: bool = True
    stores_node: bool = True

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = "llamalection",
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "IP",
        consistency_level: str = "Strong",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`pymilvus` package not found, please run `pip install pymilvus`"
        )
        try:
            import pymilvus  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        from pymilvus import MilvusClient

        self.collection_name = collection_name
        self.dim = dim
        self.embedding_field = embedding_field
        self.doc_id_field = doc_id_field
        self.consistency_level = consistency_level
        self.overwrite = overwrite

        # Select the similarity metric
        if similarity_metric.lower() in ("ip"):
            self.similarity_metric = "IP"
        elif similarity_metric.lower() in ("l2", "euclidean"):
            self.similarity_metric = "L2"

        # Connect to Milvus instance
        self.milvusclient = MilvusClient(
            uri=uri,
            token=token,
        )

        # Delete previous collection if overwriting
        if self.overwrite and self.collection_name in self.client.list_collections():
            self.milvusclient.drop_collection(self.collection_name)

        # Create the collection if it does not exist
        if self.collection_name not in self.client.list_collections():
            if self.dim is None:
                raise ValueError("Dim argument required for collection creation.")
            self.milvusclient.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                primary_field_name=MILVUS_ID_FIELD,
                vector_field_name=self.embedding_field,
                id_type="string",
                metric_type=self.similarity_metric,
                max_length=65_535,
                consistency_level=self.consistency_level,
            )

        logger.debug(f"Successfully created a new collection: {self.collection_name}")

    @property
    def client(self) -> Any:
        """Get client."""
        return self.milvusclient

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            embedding_results (List[NodeWithEmbedding]): The embeddings and their data
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        insert_list = []
        insert_ids = []

        # Process that data we are going to insert
        for result in embedding_results:
            entry = node_to_metadata_dict(result.node)
            entry[MILVUS_ID_FIELD] = result.id
            entry[self.embedding_field] = result.embedding
            insert_ids.append(result.id)
            insert_list.append(entry)

        # Insert the data into milvus
        self.milvusclient.insert(self.collection_name, insert_list)
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
        if type(ref_doc_id) != list:
            doc_ids = [ref_doc_id]
        else:
            doc_ids = ref_doc_id  # type: ignore

        # Begin by querying for the primary keys to delete
        doc_ids = ['"' + entry + '"' for entry in doc_ids]
        entries = self.milvusclient.query(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} in [{','.join(doc_ids)}]",
        )
        ids = [entry["id"] for entry in entries]
        self.milvusclient.delete(collection_name=self.collection_name, pks=ids)
        logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")

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
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = []
        output_fields = ["*"]

        # Parse the filter
        if query.filters is not None:
            expr.extend(_to_milvus_filter(query.filters))

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        if query.output_fields is not None:
            output_fields = query.output_fields

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = " and ".join(expr)

        # Perfom the search
        res = self.milvusclient.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
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
            node = metadata_dict_to_node(
                {"_node_content": hit["entity"].get("_node_content", None)}
            )
            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
