import logging
import math
from typing import Any, List

from llama_index.constants import DEFAULT_EMBEDDING_DIM
from llama_index.schema import BaseNode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

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
    flat_metadata = False

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
        return

    def _to_vecs_filters(self, filters: MetadataFilters) -> Any:
        """Convert llama filters to vecs filters. $eq is the only supported operator."""
        vecs_filter = {}
        for f in filters.filters:
            vecs_filter[f.key] = {"$eq": f.value}
        return vecs_filter

    def add(self, nodes: List[BaseNode]) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if self._collection is None:
            raise ValueError("Collection not initialized")

        data = []
        ids = []

        for node in nodes:
            # NOTE: keep text in metadata dict since there's no special field in
            #       Supabase Vector.
            metadata_dict = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=self.flat_metadata
            )

            data.append((node.node_id, node.get_embedding(), metadata_dict))
            ids.append(node.node_id)

        self._collection.upsert(records=data)

        return ids

    def get_by_id(self, doc_id: str) -> list:
        """Get row ids by doc id.

        Args:
            doc_id (str): document id
        """
        filters = {"doc_id": {"$eq": doc_id}}

        return self._collection.query(
            data=None,
            filters=filters,
            include_value=False,
            include_metadata=False,
        )

        # NOTE: list of row ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc.

        Args:
            :param ref_doc_id (str): document id

        """
        row_ids = self.get_by_id(ref_doc_id)

        if len(row_ids) > 0:
            self._collection.delete(row_ids)

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
            data=query.query_embedding,
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

            try:
                node = metadata_dict_to_node(metadata)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )
                node = TextNode(
                    id_=id_,
                    text=text,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)
            similarities.append(1.0 - math.exp(-distance))
            ids.append(id_)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
