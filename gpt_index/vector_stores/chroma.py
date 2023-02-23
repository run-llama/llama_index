"""Chroma vector store."""
import logging
import math
from typing import Any, List, cast

from gpt_index.data_structs.data_structs import Node
from gpt_index.utils import truncate_text
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


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
            raise ValueError(import_err_msg)
        from chromadb.api.models.Collection import Collection

        self._collection = cast(Collection, chroma_collection)

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {}

    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        embeddings = []
        metadatas = []
        ids = []
        documents = []
        for result in embedding_results:
            embeddings.append(result.embedding)
            metadatas.append({"document_id": result.doc_id})
            ids.append(result.id)
            documents.append(result.node.get_text())

        self._collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        self._collection.delete(where={"document_id": doc_id})

    @property
    def client(self) -> Any:
        """Return client."""
        return self._collection

    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        results = self._collection.query(
            query_embeddings=query_embedding, n_results=similarity_top_k
        )

        logging.debug(f"> Top {len(results['documents'])} nodes:")
        nodes = []
        similarities = []
        ids = []
        for result in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            node_id = result[0][0]
            node = Node(
                doc_id=node_id,
                text=result[1][0],
                extra_info=result[2][0],
                ref_doc_id=result[2][0]["document_id"],
            )
            nodes.append(node)

            similarity_score = 1.0 - math.exp(-result[3][0])
            similarities.append(similarity_score)

            logging.debug(
                f"> [Node {result[0][0]}] [Similarity score: {similarity_score}] "
                f"{truncate_text(str(result[1][0]), 100)}"
            )
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
