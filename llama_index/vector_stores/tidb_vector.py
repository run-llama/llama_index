import logging
from typing import Any, Dict, List, Optional

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

_logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "llama_index_tidb_vector"


class TiDBVector(BasePydanticVectorStore):
    stores_text = True
    flat_metadata = False

    _connection_string: str = PrivateAttr()
    _engine_args: Dict[str, Any] = PrivateAttr()
    _tidb: Any = PrivateAttr()

    def __init__(
        self,
        connection_string: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._engine_args = engine_args or {}
        try:
            from tidb_vector.integrations.vectorstore import TiDBCollection
        except ImportError:
            raise ImportError(
                "Could not import tidbvec python package. "
                "Please install it with `pip install tidbvec`."
            )

        self._tidb = TiDBCollection(
            connection_string=connection_string,
            collection_name=collection_name,
            engine_args=engine_args,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @property
    def client(self) -> Any:
        """Get client."""
        return self._tidb

    def drop_collection(self) -> None:
        """
        Drop the collection from the TiDB database.
        """
        self._tidb.drop_collection()

    @classmethod
    def class_name(cls) -> str:
        return "TiDBVector"

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids = []
        metadatas = []
        embeddings = []
        texts = []
        for node in nodes:
            ids.append(node.node_id)
            metadatas.append(node_to_metadata_dict(node, remove_text=True))
            embeddings.append(node.get_embedding())
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))

        self._tidb.insert(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            **add_kwargs,
        )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        delete_kwargs["filter"] = {"doc_id": ref_doc_id}
        self._tidb.delete(**delete_kwargs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("Query embedding must be provided.")
        return self._similarity_search_with_score(
            query.query_embedding,
            query.similarity_top_k,
            query.filters,
            **kwargs,
        )

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        filters = self._to_tidb_filters(metadata_filters)
        results = self._tidb.query(
            query_vector=embedding, k=limit, filter=filters, **kwargs
        )

        nodes = []
        similarities = []
        ids = []
        for row in results:
            try:
                node = metadata_dict_to_node(row.metadata)
                node.set_content(str(row.document))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                _logger.warning(
                    "Failed to parse metadata dict, falling back to legacy logic."
                )
                node = TextNode(
                    id_=row.id,
                    text=row.document,
                    metadata=row.metadata,
                )
            similarities.append((1 - row.distance) if row.distance is not None else 0)
            ids.append(row.id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _to_tidb_filters(
        self, metadata_filters: Optional[MetadataFilters] = None
    ) -> Optional[Dict[str, Any]]:
        if metadata_filters is None:
            return None

        condition = "$and"
        if metadata_filters.condition == FilterCondition.OR:
            condition = "$or"

        filters = []
        for filter in metadata_filters.filters:
            if filter.operator == FilterOperator.EQ:
                filters.append({filter.key: {"$eq": filter.value}})
            elif filter.operator == FilterOperator.NE:
                filters.append({filter.key: {"$ne": filter.value}})
            elif filter.operator == FilterOperator.GT:
                filters.append({filter.key: {"$gt": filter.value}})
            elif filter.operator == FilterOperator.GTE:
                filters.append({filter.key: {"$gte": filter.value}})
            elif filter.operator == FilterOperator.LT:
                filters.append({filter.key: {"$lt": filter.value}})
            elif filter.operator == FilterOperator.LTE:
                filters.append({filter.key: {"$lte": filter.value}})
            elif filter.operator == FilterOperator.IN:
                filters.append({filter.key: {"$in": filter.value}})
            elif filter.operator == FilterOperator.NIN:
                filters.append({filter.key: {"$nin": filter.value}})
            else:
                raise ValueError(f"Unsupported operator: {filter.operator}")

        return {condition: filters}
