"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
import logging
from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.utils import iter_batch
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
    """

    stores_text: bool = True

    def __init__(
        self, collection_name: str, client: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if client is None:
            raise ValueError("Missing Qdrant client!")

        self._client = cast(qdrant_client.QdrantClient, client)
        self._collection_name = collection_name
        self._collection_initialized = self._collection_exists(collection_name)

        self._batch_size = kwargs.get("batch_size", 100)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        if "client" not in config_dict:
            raise ValueError("Missing Qdrant client!")
        return cls(**config_dict)

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "collection_name": self._collection_name,
        }

    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        from qdrant_client.http import models as rest

        if len(embedding_results) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self._collection_name,
                vector_size=len(embedding_results[0].embedding),
            )

        ids = []
        for result_batch in iter_batch(embedding_results, self._batch_size):
            new_ids = []
            vectors = []
            payloads = []
            for result in result_batch:
                new_ids.append(result.id)
                vectors.append(result.embedding)
                node = result.node
                payloads.append(
                    {
                        "doc_id": result.doc_id,
                        "text": node.get_text(),
                        "extra_info": node.extra_info,
                    }
                )

            self._client.upsert(
                collection_name=self._collection_name,
                points=rest.Batch(
                    ids=new_ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            ids.extend(new_ids)
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id: (str): document id

        """
        from qdrant_client.http import models as rest

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    )
                ]
            ),
        )

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._client

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.http import models as rest

        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        self._collection_initialized = True

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        from grpc import RpcError
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            self._client.get_collection(collection_name)
        except (RpcError, UnexpectedResponse):
            return False
        return True

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by

        """
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchValue,
            Payload,
        )

        query_embedding = cast(List[float], query.query_embedding)

        response = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=cast(int, query.similarity_top_k),
            query_filter=None
            if not query.doc_ids
            else Filter(
                must=[
                    Filter(
                        should=[
                            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                            for doc_id in query.doc_ids
                        ],
                    )
                ]
            ),
        )

        logger.debug(f"> Top {len(response)} nodes:")

        nodes = []
        similarities = []
        ids = []
        for point in response:
            payload = cast(Payload, point.payload)
            node = Node(
                doc_id=str(point.id),
                text=payload.get("text"),
                extra_info=payload.get("extra_info"),
                relationships={
                    DocumentRelationship.SOURCE: payload.get("doc_id", "None"),
                },
            )
            nodes.append(node)
            similarities.append(point.score)
            ids.append(str(point.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
