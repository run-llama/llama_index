"""DynamoDB vector store index."""
from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, cast

from llama_index.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
)
from llama_index.schema import BaseNode
from llama_index.storage.kvstore.dynamodb_kvstore import DynamoDBKVStore
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

logger = getLogger(__name__)

DEFAULT_NAMESPACE = "vector_store"

LEARNER_MODES = {
    VectorStoreQueryMode.SVM,
    VectorStoreQueryMode.LINEAR_REGRESSION,
    VectorStoreQueryMode.LOGISTIC_REGRESSION,
}


class DynamoDBVectorStore(VectorStore):
    """DynamoDB Vector Store.

    In this vector store, embeddings are stored within dynamodb table.
    This class was implemented with reference to SimpleVectorStore.

    Args:
        dynamodb_kvstore (DynamoDBKVStore): data store
        namespace (Optional[str]): namespace
    """

    stores_text: bool = False

    def __init__(
        self, dynamodb_kvstore: DynamoDBKVStore, namespace: str | None = None
    ) -> None:
        """Initialize params."""
        self._kvstore = dynamodb_kvstore
        namespace = namespace or DEFAULT_NAMESPACE
        self._collection_embedding = f"{namespace}/embedding"
        self._collection_text_id_to_doc_id = f"{namespace}/text_id_to_doc_id"
        self._key_value = "value"

    @classmethod
    def from_table_name(
        cls, table_name: str, namespace: str | None = None
    ) -> DynamoDBVectorStore:
        """Load from DynamoDB table name."""
        dynamodb_kvstore = DynamoDBKVStore.from_table_name(table_name=table_name)
        return cls(dynamodb_kvstore=dynamodb_kvstore, namespace=namespace)

    @property
    def client(self) -> None:
        """Get client."""
        return

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        item = self._kvstore.get(key=text_id, collection=self._collection_embedding)
        item = cast(Dict[str, List[float]], item)
        return item[self._key_value]

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index."""
        response = []
        for node in nodes:
            self._kvstore.put(
                key=node.node_id,
                val={self._key_value: node.get_embedding()},
                collection=self._collection_embedding,
            )
            self._kvstore.put(
                key=node.node_id,
                val={self._key_value: node.ref_doc_id},
                collection=self._collection_text_id_to_doc_id,
            )
            response.append(node.node_id)
        return response

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        text_ids_to_delete = set()
        for text_id, item in self._kvstore.get_all(
            collection=self._collection_text_id_to_doc_id
        ).items():
            if ref_doc_id == item[self._key_value]:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            self._kvstore.delete(key=text_id, collection=self._collection_embedding)
            self._kvstore.delete(
                key=text_id, collection=self._collection_text_id_to_doc_id
            )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Get nodes for response."""
        if query.filters is not None:
            raise ValueError(
                "Metadata filters not implemented for SimpleVectorStore yet."
            )

        # TODO: consolidate with get_query_text_embedding_similarities
        items = self._kvstore.get_all(collection=self._collection_embedding).items()

        if query.node_ids:
            available_ids = set(query.node_ids)

            node_ids = [k for k, _ in items if k in available_ids]
            embeddings = [v[self._key_value] for k, v in items if k in available_ids]
        else:
            node_ids = [k for k, _ in items]
            embeddings = [v[self._key_value] for k, v in items]

        query_embedding = cast(List[float], query.query_embedding)
        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding=query_embedding,
                embeddings=embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding=query_embedding,
                embeddings=embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
