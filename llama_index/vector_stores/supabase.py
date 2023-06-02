import math
from typing import Any, List

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class SupabaseVectorStore(VectorStore):
    """Supbabase vector store.

    In this vector store, embeddings are stored within a pg-vector collection.

    During query time, the index uses pg-vector/supabase to query for the top
    k most similar nodes.

    Args:
        postgres_connection_string (str):
            postgres connection string

        collection_name (str):
            name of the collection to store the embeddings in

    """

    stores_text = True

    def __init__(
        self, postgres_connection_string: str, collection_name: str, **kwargs: Any
    ) -> None:
        """Init params."""
        import_err_msg = "`vecs` package not found, please run `pip install vecs`"
        try:
            import vecs
        except ImportError:
            raise ImportError(import_err_msg)

        client = vecs.create_client(postgres_connection_string)

        self._collection = client.get_collection(name=collection_name)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def _to_vecs_filters(self, filters: MetadataFilters) -> Any:
        """Convert llama filters to vecs filters. $eq is the only supported operator."""
        vecs_filter = {}
        for f in filters:
            vecs_filter[f.metadata_key] = {"$eq": f.metadata_value}
        return vecs_filter

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        data = []
        ids = []
        for result in embedding_results:
            data.append((result.id, result.embedding, result.node.text))
            ids.append(result.id)

        self._collection.upsert(vectors=data)

        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        raise NotImplementedError("Delete not yet implemented for vecs.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (List[float]): query embedding

        """
        filters = None
        if query.filters is not None:
            filters = self._to_vecs_filters(query.filters)

        result = self._collection.query(
            query_vector=query.query_embedding,
            limit=query.similarity_top_k,
            filters=filters,
            include_value=True,
            include_metadata=True,
        )

        similarities = []
        ids = []
        nodes = []
        for r in result:
            """shape of the result is [(vector, distance)]"""
            distance = r[1]
            similarities.append(1.0 - math.exp(-distance))
            metadata = r[2]

            id = r[0]
            ids.append(id)

            node = Node(
                doc_id=id,
                text=metadata,
                relationships={
                    DocumentRelationship.SOURCE: id,
                },
            )

            nodes.append(node)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
