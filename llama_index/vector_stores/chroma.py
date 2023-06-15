"""Chroma vector store."""
import logging
import math
from typing import Any, List, Tuple, cast

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.utils import truncate_text
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

logger = logging.getLogger(__name__)


def _to_chroma_filter(standard_filters: MetadataFilters) -> dict:
    """Translate standard metadata filters to Chroma specific spec."""
    filters = {}
    for filter in standard_filters.filters:
        filters[filter.key] = filter.value
    return filters


def _legacy_metadata_dict_to_node(metadata: dict) -> Tuple[dict, dict, dict]:
    extra_info = metadata
    node_info: dict = {}
    relationships = {
        DocumentRelationship.SOURCE: metadata["document_id"],
    }
    return extra_info, node_info, relationships


class ChromaVectorStore(VectorStore):
    """Chroma vector store.

    In this vector store, embeddings are stored within a ChromaDB collection.

    During query time, the index uses ChromaDB to query for the top
    k most similar nodes.

    Args:
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    """

    stores_text: bool = True

    def __init__(self, chroma_collection: Any, **kwargs: Any) -> None:
        """Init params."""
        import_err_msg = (
            "`chromadb` package not found, please run `pip install chromadb`"
        )
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)
        from chromadb.api.models.Collection import Collection

        self._collection = cast(Collection, chroma_collection)

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        embeddings = []
        metadatas = []
        ids = []
        documents = []
        for result in embedding_results:
            embeddings.append(result.embedding)
            metadatas.append(node_to_metadata_dict(result.node))
            ids.append(result.id)
            documents.append(result.node.text or "")

        self._collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._collection.delete(where={"document_id": ref_doc_id})

    @property
    def client(self) -> Any:
        """Return client."""
        return self._collection

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify metadata filters via both query and kwargs. "
                    "Use kwargs only for chroma specific items that are "
                    "not supported via the generic query interface."
                )
            where = _to_chroma_filter(query.filters)
        else:
            where = kwargs.pop("where", {})

        results = self._collection.query(
            query_embeddings=query.query_embedding,
            n_results=query.similarity_top_k,
            where=where,
            **kwargs,
        )

        logger.debug(f"> Top {len(results['documents'])} nodes:")
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
                extra_info, node_info, relationships = metadata_dict_to_node(metadata)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                extra_info, node_info, relationships = _legacy_metadata_dict_to_node(
                    metadata
                )

            node = Node(
                doc_id=node_id,
                text=text,
                extra_info=extra_info,
                node_info=node_info,
                relationships=relationships,
            )
            nodes.append(node)

            similarity_score = 1.0 - math.exp(-distance)
            similarities.append(similarity_score)

            logger.debug(
                f"> [Node {node_id}] [Similarity score: {similarity_score}] "
                f"{truncate_text(str(text), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
