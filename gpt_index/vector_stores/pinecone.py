"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

import os
from typing import Any, Dict, List, Optional, cast

from gpt_index.data_structs.node_v2 import Node, DocumentRelationship
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)
import logging

_logger = logging.getLogger(__name__)


def get_metadata_from_node_info(
    node_info: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get metadata from node extra info."""
    metadata = {}
    for key, value in node_info.items():
        metadata[field_prefix + "_" + key] = value
    return metadata


def get_node_info_from_metadata(
    metadata: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get node extra info from metadata."""
    node_extra_info = {}
    for key, value in metadata.items():
        if key.startswith(field_prefix + "_"):
            node_extra_info[key.replace(field_prefix + "_", "")] = value
    return node_extra_info


class PineconeVectorStore(VectorStore):
    """Pinecone Vector Store.

    In this vector store, embeddings and docs are stored within a
    Pinecone index.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes.

    Args:
        pinecone_index (Optional[pinecone.Index]): Pinecone index instance
        pinecone_kwargs (Optional[Dict]): kwargs to pass to Pinecone index.
            NOTE: deprecated. If specified, then insert_kwargs, query_kwargs,
            and delete_kwargs cannot be specified.
        insert_kwargs (Optional[Dict]): insert kwargs during `upsert` call.
        query_kwargs (Optional[Dict]): query kwargs during `query` call.
        delete_kwargs (Optional[Dict]): delete kwargs during `delete` call.

    """

    stores_text: bool = True

    def __init__(
        self,
        pinecone_index: Optional[Any] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        pinecone_kwargs: Optional[Dict] = None,
        insert_kwargs: Optional[Dict] = None,
        query_kwargs: Optional[Dict] = None,
        delete_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`pinecone` package not found, please run `pip install pinecone-client`"
        )
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._index_name = index_name
        self._environment = environment
        if pinecone_index is not None:
            self._pinecone_index = cast(pinecone.Index, pinecone_index)
            _logger.warn(
                "If directly passing in client, cannot automatically reconstruct "
                "connetion after save_to_disk/load_from_disk."
                "For automatic reload, store PINECONE_API_KEY in env variable and "
                "pass in index_name and environment instead."
            )
        else:
            if "PINECONE_API_KEY" not in os.environ:
                raise ValueError(
                    "Must specify PINECONE_API_KEY via env variable "
                    "if not directly passing in client."
                )
            if index_name is None or environment is None:
                raise ValueError(
                    "Must specify index_name and environment "
                    "if not directly passing in client."
                )

            pinecone.init(environment=environment)
            self._pinecone_index = pinecone.Index(index_name)

        self._metadata_filters = metadata_filters or {}
        self._pinecone_kwargs = pinecone_kwargs or {}
        if pinecone_kwargs and (insert_kwargs or query_kwargs or delete_kwargs):
            raise ValueError(
                "pinecone_kwargs cannot be specified if insert_kwargs, "
                "query_kwargs, or delete_kwargs are specified."
            )
        elif pinecone_kwargs:
            self._insert_kwargs = pinecone_kwargs
            self._query_kwargs = pinecone_kwargs
            self._delete_kwargs = pinecone_kwargs
        else:
            self._insert_kwargs = insert_kwargs or {}
            self._query_kwargs = query_kwargs or {}
            self._delete_kwargs = delete_kwargs or {}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        return cls(**config_dict)

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "index_name": self._index_name,
            "environment": self._environment,
            "metadata_filters": self._metadata_filters,
            "pinecone_kwargs": self._pinecone_kwargs,
            "insert_kwargs": self._insert_kwargs,
            "query_kwargs": self._query_kwargs,
            "delete_kwargs": self._delete_kwargs,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        ids = []
        for result in embedding_results:
            new_id = result.id
            node = result.node
            text_embedding = result.embedding

            metadata = {
                "text": node.get_text(),
                # NOTE: this is the reference to source doc
                "doc_id": result.doc_id,
                "id": new_id,
            }
            if node.extra_info:
                # TODO: check if overlap with default metadata keys
                metadata.update(
                    get_metadata_from_node_info(node.extra_info, "extra_info")
                )
            if node.node_info:
                # TODO: check if overlap with default metadata keys
                metadata.update(
                    get_metadata_from_node_info(node.node_info, "node_info")
                )
            # if additional metadata keys overlap with the default keys,
            # then throw an error
            intersecting_keys = set(metadata.keys()).intersection(
                self._metadata_filters.keys()
            )
            if intersecting_keys:
                raise ValueError(
                    "metadata_filters keys overlap with default "
                    f"metadata keys: {intersecting_keys}"
                )
            metadata.update(self._metadata_filters)
            self._pinecone_index.upsert(
                [(new_id, text_embedding, metadata)], **self._pinecone_kwargs
            )
            ids.append(new_id)
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}, **self._pinecone_kwargs
        )

    @property
    def client(self) -> Any:
        """Return Pinecone client."""
        return self._pinecone_index

    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int,
        doc_ids: Optional[List[str]] = None,
        query_str: Optional[str] = None,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        response = self._pinecone_index.query(
            query_embedding,
            top_k=similarity_top_k,
            include_values=True,
            include_metadata=True,
            filter=self._metadata_filters,
            **self._pinecone_kwargs,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for match in response.matches:
            text = match.metadata["text"]
            extra_info = get_node_info_from_metadata(match.metadata, "extra_info")
            node_info = get_node_info_from_metadata(match.metadata, "node_info")
            doc_id = match.metadata["doc_id"]
            id = match.metadata["id"]

            node = Node(
                text=text,
                extra_info=extra_info,
                node_info=node_info,
                doc_id=id,
                relationships={DocumentRelationship.SOURCE: doc_id},
            )
            top_k_ids.append(match.id)
            top_k_nodes.append(node)
            top_k_scores.append(match.score)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
