"""MongoDB Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
    legacy_metadata_dict_to_node,
)

logger = logging.getLogger(__name__)


def _to_mongodb_filter(standard_filters: MetadataFilters) -> Dict:
    """Convert from standard dataclass to filter dict."""
    filters = {}
    for filter in standard_filters.filters:
        filters[filter.key] = filter.value
    return filters


class MongoDBAtlasVectorSearch(VectorStore):
    """MongoDB Vector Store.

    In this vector store, embeddings and docs are stored within a
    MongoDB index.

    During query time, the index uses Atlas knnbeta to query for the top
    k most similar nodes.

    Args:
        mongodb_index (Optional[pymongo.MongoClient]): MongoDB index instance
        db_name (str): MongoDB database name
        collection_name (str): MongoDB collection name
        insert_kwargs (Optional[Dict]): kwargs used during `insert`.

    """

    stores_text: bool = True

    def __init__(
        self,
        mongodb_client: Optional[Any] = None,
        db_name: str = "default_db",
        collection_name: str = "default_collection",
        index_name: str = "default",
        id_key: str = "id",
        embedding_key: str = "embedding",
        text_key: str = "text",
        metadata_key: str = "metadata",
        insert_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = "`pymongo` package not found, please run `pip install pymongo`"
        try:
            import pymongo  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if mongodb_client is not None:
            self._mongodb_client = cast(pymongo.MongoClient, mongodb_client)
        else:
            if "MONGO_URI" not in os.environ:
                raise ValueError(
                    "Must specify MONGO_URI via env variable "
                    "if not directly passing in client."
                )
            self._mongodb_client = pymongo.MongoClient(os.environ["MONGO_URI"])

        self._collection = self._mongodb_client[db_name][collection_name]
        self._index_name = index_name
        self._embedding_key = embedding_key
        self._id_key = id_key
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._insert_kwargs = insert_kwargs or {}

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        ids = []
        data_to_insert = []
        for result in embedding_results:
            node_id = result.id
            node = result.node

            metadata = node_to_metadata_dict(node, remove_text=True)

            entry = {
                self._id_key: node_id,
                self._embedding_key: result.embedding,
                self._text_key: node.get_content(metadata_mode=MetadataMode.NONE) or "",
                self._metadata_key: metadata,
            }
            data_to_insert.append(entry)
            ids.append(node_id)
        logger.debug("Inserting data into MongoDB: %s", data_to_insert)
        insert_result = self._collection.insert_many(
            data_to_insert, **self._insert_kwargs
        )
        logger.debug("Result of insert: %s", insert_result)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # delete by filtering on the doc_id metadata
        self._collection.delete_one(
            filter={self._metadata_key + ".ref_doc_id": ref_doc_id}, **delete_kwargs
        )

    @property
    def client(self) -> Any:
        """Return MongoDB client."""
        return self._mongodb_client

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """

        knn_beta: Dict[str, Any] = {
            "vector": query.query_embedding,
            "path": self._embedding_key,
            "k": query.similarity_top_k,
        }
        if query.filters:
            knn_beta["filter"] = _to_mongodb_filter(query.filters)

        pipeline = [
            {
                "$search": {
                    "index": self._index_name,
                    "knnBeta": knn_beta,
                }
            },
            {"$project": {"score": {"$meta": "searchScore"}, self._embedding_key: 0}},
        ]
        logger.debug("Running query pipeline: %s", pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            id = res.pop(self._id_key)

            try:
                node = metadata_dict_to_node(res.pop(self._metadata_key))
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    res.pop(self._metadata_key)
                )

                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    node_info=node_info,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            top_k_ids.append(id)
            top_k_nodes.append(node)
            top_k_scores.append(score)
        result = VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
        logger.debug("Result of query: %s", result)
        return result
