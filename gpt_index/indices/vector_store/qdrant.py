"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
import logging
from typing import Any, List, Optional, cast

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.vector_store.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)
from gpt_index.utils import get_new_id


class QdrantVectorStore(VectorStore):
    """GPT Qdrant Index.

    The GPTQdrantIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Qdrant collection.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Qdrant.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection
    """

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if client is None:
            raise ValueError("client cannot be None.")

        self._client = cast(qdrant_client.QdrantClient, client)
        self._collection_name = collection_name
        self._collection_initialized = self._collection_exists(collection_name)
    
    @property
    def config_dict(self) -> dict:
        return {
            'collection_name': self._collection_name,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult]
    ) -> None:
        """Add document to index."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        for result in embedding_results:
            new_id = result.id
            node = result.node
            text_embedding = result.embedding
            collection_name = self._collection_name
            # assign a new_id if current_id conflicts with existing ids
            while True:
                try:
                    self._client.http.points_api.get_point(
                        collection_name=collection_name, id=new_id
                    )
                except UnexpectedResponse:
                    break
                new_id = get_new_id(set())

            # Create the Qdrant collection, if it does not exist yet
            if not self._collection_initialized:
                self._create_collection(
                    collection_name=collection_name,
                    vector_size=len(text_embedding),
                )
                self._collection_initialized = True

            payload = {
                "doc_id": result.doc_id,
                "text": node.get_text(),
                "index": node.index,
            }

            self._client.upsert(
                collection_name=collection_name,
                points=[
                    rest.PointStruct(
                        id=new_id,
                        vector=text_embedding,
                        payload=payload,
                    )
                ],
            )

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
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

    def _collection_exists(self, collection_name: str) -> bool:
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            response = self._client.http.collections_api.get_collection(collection_name)
            return response.result is not None
        except UnexpectedResponse:
            return False

    def query(self, 
        query_embedding: List[float], 
        similarity_top_k: int, 
    ) -> VectorStoreQueryResult:
        from qdrant_client.http.models.models import Payload

        response = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=cast(int, similarity_top_k),
        )

        logging.debug(f"> Top {len(response)} nodes:")

        nodes = []
        similarities = []
        for point in response:
            payload = cast(Payload, point.payload)
            node = Node(
                ref_doc_id=payload.get("doc_id"),
                text=payload.get("text"),
            )
            nodes.append(node)
            similarities.append(point.score)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities)
