import logging
import math
from typing import Any, List

from llama_index.constants import DEFAULT_EMBEDDING_DIM
from llama_index.data_structs.node import Node
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

logger = logging.getLogger(__name__)


class SupabaseVectorStore(VectorStore):
    """Supbabase Vector.

    In this vector store, embeddings are stored in Postgres table using pgvector.

    During query time, the index uses pgvector/Supabase to query for the top
    k most similar nodes.

    Args:
        postgres_connection_string (str):
            postgres connection string

        collection_name (str):
            name of the collection to store the embeddings in

    """

    stores_text = True

    def __init__(
        self,
        postgres_connection_string: str,
        collection_name: str,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import_err_msg = "`vecs` package not found, please run `pip install vecs`"
        try:
            import vecs
            from vecs.collection import CollectionNotFound
        except ImportError:
            raise ImportError(import_err_msg)

        client = vecs.create_client(postgres_connection_string)

        try:
            self._collection = client.get_collection(name=collection_name)
        except CollectionNotFound:
            logger.info(
                f"Collection {collection_name} does not exist, "
                f"try creating one with dimension={dimension}"
            )
            self._collection = client.create_collection(
                name=collection_name, dimension=dimension
            )

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def _to_vecs_filters(self, filters: MetadataFilters) -> Any:
        """Convert llama filters to vecs filters. $eq is the only supported operator."""
        vecs_filter = {}
        for f in filters.filters:
            vecs_filter[f.key] = {"$eq": f.value}
        return vecs_filter

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        if self._collection is None:
            raise ValueError("Collection not initialized")

        data = []
        ids = []

        for result in embedding_results:
            metadata_dict = node_to_metadata_dict(result.node)
            # NOTE: keep text in metadata dict since there's no special field in
            #       Supabase Vector.
            metadata_dict["text"] = result.node.text
            data.append((result.id, result.embedding, metadata_dict))
            ids.append(result.id)

        self._collection.upsert(vectors=data)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc.

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

        results = self._collection.query(
            query_vector=query.query_embedding,
            limit=query.similarity_top_k,
            filters=filters,
            include_value=True,
            include_metadata=True,
        )

        similarities = []
        ids = []
        nodes = []
        for id_, distance, metadata in results:
            """shape of the result is [(vector, distance, metadata)]"""

            text = metadata.pop("text", None)
            extra_info, node_info, relationships = metadata_dict_to_node(metadata)
            node = Node(
                doc_id=id_,
                text=text,
                extra_info=extra_info,
                node_info=node_info,
                relationships=relationships,
            )

            nodes.append(node)
            similarities.append(1.0 - math.exp(-distance))
            ids.append(id_)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
