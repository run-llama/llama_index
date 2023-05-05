"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
import logging
from typing import Any, List, Optional, cast

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.utils import iter_batch
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
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

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        from qdrant_client.http import models as rest

        if len(embedding_results) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self._collection_name,
                vector_size=len(embedding_results[0].embedding),
            )

        ids = []
        for result_batch in iter_batch(embedding_results, self._batch_size):
            node_ids = []
            vectors = []
            payloads = []
            for result in result_batch:
                node_ids.append(result.id)
                vectors.append(result.embedding)
                node = result.node
                payloads.append(
                    {
                        "doc_id": result.ref_doc_id,
                        "text": node.get_text(),
                        "extra_info": node.extra_info,
                    }
                )

            self._client.upsert(
                collection_name=self._collection_name,
                points=rest.Batch.construct(
                    ids=node_ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            ids.extend(node_ids)
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
        except (RpcError, UnexpectedResponse, ValueError):
            return False
        return True

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
        """
        from qdrant_client.http.models import Payload, Filter

        query_embedding = cast(List[float], query.query_embedding)

        response = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=cast(int, query.similarity_top_k),
            query_filter=cast(Filter, self._build_query_filter(query)),
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

    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[Any]:
        if not query.doc_ids and not query.query_str:
            return None

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
        )

        must_conditions = []

        if query.doc_ids:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=[doc_id for doc_id in query.doc_ids]),
                )
            )
        return Filter(must=must_conditions)
