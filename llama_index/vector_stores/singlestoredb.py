from typing import Any, List, Optional, Dict
from singlestoredb import connect
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)

class SingleStoreVectorStore(BasePydanticVectorStore):
    """SingleStore vector store.

    This vector store stores embeddings within a SingleStore database table.

    During query time, the index uses SingleStore to query for the top
    k most similar nodes.

    Args:
        db_config (dict): Configuration for connecting to SingleStore database
        table_name (str): Name of the table to store the vector data
    """

    stores_text: bool = True
    flat_metadata: bool = True

    db_config: Dict[str, Any]
    table_name: str

    def __init__(self, db_config: Dict[str, Any], table_name: str = "vector_store", **kwargs: Any) -> None:
        """Init params."""
        self.db_config = db_config
        self.table_name = table_name
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "SingleStoreVectorStore"

    def _get_connection(self) -> Any:
        return connect(**self.db_config)

    def add(self, nodes: List[BaseNode]) -> List[str]:
        """Add nodes to index.

        Args
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            for node in nodes:
                embedding = node.get_embedding()
                metadata = node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
                cursor.execute(
                    f"INSERT INTO {self.table_name} (id, vector, metadata) VALUES (%s, %s, %s)",
                    (node.node_id, embedding, metadata)
                )
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (ref_doc_id,))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        query_embedding = query.query_embedding
        similarity_top_k = query.similarity_top_k

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT * FROM {self.table_name} ORDER BY "
                "DOT_PRODUCT(vector, %s) DESC LIMIT %s",
                (query_embedding, similarity_top_k)
            )
            results = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

        nodes = []
        similarities = []
        ids = []
        for result in results:
            node_id, vector, metadata = result
            node = metadata_dict_to_node(metadata)
            node.set_embedding(vector)

            nodes.append(node)
            similarity_score = cosine_similarity(query_embedding, vector)
            similarities.append(similarity_score)
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
