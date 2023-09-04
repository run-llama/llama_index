"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
import logging
from typing import Any, List, Optional, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.schema import TextNode
from llama_index.utils import iter_batch
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
    legacy_metadata_dict_to_node,
)

logger = logging.getLogger(__name__)
import_err_msg = (
    "`qdrant-client` package not found, please run `pip install qdrant-client`"
)


class QdrantVectorStore(BasePydanticVectorStore):
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
    flat_metadata: bool = False

    collection_name: str
    url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    client_kwargs: dict = Field(default_factory=dict)

    _client: Any = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        client_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if client is None:
            raise ValueError("Missing Qdrant client!")

        self._client = cast(qdrant_client.QdrantClient, client)
        self._collection_initialized = self._collection_exists(collection_name)

        super().__init__(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            client_kwargs=client_kwargs or {},
        )

    @classmethod
    def from_params(
        cls,
        collection_name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        client_kwargs: Optional[dict] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> "QdrantVectorStore":
        """Create a connection to a remote Qdrant vector store from a config."""
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        client_kwargs = client_kwargs or {}
        return cls(
            collection_name=collection_name,
            client=qdrant_client.QdrantClient(
                url=url, api_key=api_key, **client_kwargs
            ),
            batch_size=batch_size,
            client_kwargs=client_kwargs,
            url=url,
            api_key=api_key,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QdraantVectorStore"

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        from qdrant_client.http import models as rest

        if len(embedding_results) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self.collection_name,
                vector_size=len(embedding_results[0].embedding),
            )

        ids = []
        for result_batch in iter_batch(embedding_results, self.batch_size):
            node_ids = []
            vectors = []
            payloads = []
            for result in result_batch:
                assert isinstance(result, NodeWithEmbedding)
                assert isinstance(result.node, TextNode)
                node_ids.append(result.id)
                vectors.append(result.embedding)
                node = result.node

                metadata = node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )

                payloads.append(metadata)

            self._client.upsert(
                collection_name=self.collection_name,
                points=rest.Batch.construct(
                    ids=node_ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            ids.extend(node_ids)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        from qdrant_client.http import models as rest

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=ref_doc_id)
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
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
        """
        from qdrant_client.http.models import Filter, Payload

        query_embedding = cast(List[float], query.query_embedding)

        response = self._client.search(
            collection_name=self.collection_name,
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
            try:
                node = metadata_dict_to_node(payload)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                logger.debug("Failed to parse Node metadata, fallback to legacy logic.")
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    payload
                )

                node = TextNode(
                    id_=str(point.id),
                    text=payload.get("text"),
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
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
            MatchValue,
            Range,
        )

        must_conditions = []

        if query.doc_ids:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=query.doc_ids),
                )
            )

        if query.node_ids:
            must_conditions.append(
                FieldCondition(
                    key="id",
                    match=MatchAny(any=query.node_ids),
                )
            )

        # Qdrant does not use the query.query_str property for the filtering. Full-text
        # filtering cannot handle longer queries and can effectively filter our all the
        # nodes. See: https://github.com/jerryjliu/llama_index/pull/1181

        if query.filters is None:
            return Filter(must=must_conditions)

        for subfilter in query.filters.filters:
            if isinstance(subfilter.value, float):
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(
                            gte=subfilter.value,
                            lte=subfilter.value,
                        ),
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchValue(value=subfilter.value),
                    )
                )

        return Filter(must=must_conditions)
