import logging
import math
from typing import Any, List, cast

from llama_index.schema import MetadataMode, TextNode
from llama_index.utils import truncate_text
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
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


def _to_bagel_filter(standard_filters: MetadataFilters) -> dict:
    """
        Translate standard metadata filters to Bagel specific spec.
    """
    filters = {}
    for filter in standard_filters.filters:
        filters[filter.key] = filter.value
    return filters


class BagelVectorStore(VectorStore):
    """
        Vector store for Bagel.
    """

    # support for Bagel specific parameters
    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(self, collection: Any, **kwargs: Any) -> None:
        """
            Initialize BagelVectorStore.

            Args:
                collection: Bagel collection.
                **kwargs: Additional arguments.
        """

        try:
            import bagel
        except ImportError:
            raise ImportError(
                "Bagel is not installed. Please install bagel using `pip install bagel`."
            )

        from bagel.api.Cluster import Cluster

        if not isinstance(collection, Cluster):
            raise ValueError(
                "collection must be a bagel Cluster. Please use bagel client to get a Cluster."
            )

        self._collection = collection

    def add(self, embedding_results: List[NodeWithEmbedding], **kwargs: Any) -> List[str]:
        """
            Add a list of nodes with embeddings to the vector store.

            Args:
                embedding_results: List of nodes with embeddings.
                kwargs: Additional arguments.

            Returns:
                List of document ids.
        """
        if not self._collection:
            raise ValueError("collection not set")

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for node_with_embedding in embedding_results:
            ids.append(node_with_embedding.id)
            embeddings.append(node_with_embedding.embedding)
            metadatas.append(node_to_metadata_dict(
                node_with_embedding.node, remove_text=True, flat_metadata=self.flat_metadata))
            documents.append(node_with_embedding.node.get_content(
                metadata_mode=MetadataMode.NONE) or "")

        self._collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

        return ids

    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """
            Delete a document from the vector store.

            Args:
                ref_doc_id: Reference document id.
                kwargs: Additional arguments.
        """
        if not self._collection:
            raise ValueError("collection not set")

        self._collection.delete(ids=[ref_doc_id])

    @property
    def client(self) -> Any:
        """
            Get the Bagel cluster.
        """
        return self._collection

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
            Query the vector store.

            Args:
                query: Query to run.
                kwargs: Additional arguments.

            Returns:
                Query result.
        """
        if not self._collection:
            raise ValueError("collection not set")

        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError("Cannot specify both filters and where")
            where = _to_bagel_filter(query.filters)
        else:
            where = kwargs.get("where", {})

        results = self._collection.find(
            query_embeddings=query.query_embedding,
            where=where,
            n_results=query.similarity_top_k,
            **kwargs,
        )

        logger.debug(f"query results: {results}")

        nodes = []
        similarities = []
        ids = []

        for node_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata
                )

                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)
            similarities.append(1.0 - math.exp(-distance))
            ids.append(node_id)

            logger.debug(f"node: {node}")
            logger.debug(f"similarity: {1.0 - math.exp(-distance)}")
            logger.debug(f"id: {node_id}")

        return VectorStoreQueryResult(
            nodes=nodes, similarities=similarities, ids=ids
        )
