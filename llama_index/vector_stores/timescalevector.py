import enum
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from llama_index.constants import DEFAULT_EMBEDDING_DIM
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict


class IndexType(enum.Enum):
    """Enumerator for the supported Index types."""

    TIMESCALE_VECTOR = 1
    PGVECTOR_IVFFLAT = 2
    PGVECTOR_HNSW = 3


class TimescaleVectorStore(VectorStore):
    stores_text = True
    flat_metadata = False

    def __init__(
        self,
        service_url: str,
        table_name: str,
        num_dimensions: int = DEFAULT_EMBEDDING_DIM,
        time_partition_interval: Optional[timedelta] = None,
    ) -> None:
        try:
            from timescale_vector import client  # noqa
        except ImportError:
            raise ImportError("`timescale-vector` package should be pre installed")

        self.service_url = service_url
        self.table_name: str = table_name.lower()
        self.num_dimensions = num_dimensions
        self.time_partition_interval = time_partition_interval

        self._create_clients()
        self._create_tables()

    async def close(self) -> None:
        self._sync_client.close()
        await self._async_client.close()

    @classmethod
    def from_params(
        cls,
        service_url: str,
        table_name: str,
        num_dimensions: int = DEFAULT_EMBEDDING_DIM,
        time_partition_interval: Optional[timedelta] = None,
    ) -> "TimescaleVectorStore":
        return cls(
            service_url=service_url,
            table_name=table_name,
            num_dimensions=num_dimensions,
            time_partition_interval=time_partition_interval,
        )

    def _create_clients(self) -> None:
        from timescale_vector import client

        # in the normal case doesn't restrict the id type to even uuid.
        # Allow arbitrary text
        id_type = "TEXT"
        if self.time_partition_interval is not None:
            # for time partitioned tables, the id type must be UUID v1
            id_type = "UUID"

        self._sync_client = client.Sync(
            self.service_url,
            self.table_name,
            self.num_dimensions,
            id_type=id_type,
            time_partition_interval=self.time_partition_interval,
        )
        self._async_client = client.Async(
            self.service_url,
            self.table_name,
            self.num_dimensions,
            id_type=id_type,
            time_partition_interval=self.time_partition_interval,
        )

    def _create_tables(self) -> None:
        self._sync_client.create_tables()

    def _node_to_row(self, node: BaseNode) -> Any:
        metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )
        # reuse the node id in the common  case
        id = node.node_id
        if self.time_partition_interval is not None:
            # for time partitioned tables, the id must be a UUID v1,
            # so generate one if it's not already set
            try:
                # Attempt to parse the UUID from the string
                parsed_uuid = uuid.UUID(id)
                if parsed_uuid.version != 1:
                    id = str(uuid.uuid1())
            except ValueError:
                id = str(uuid.uuid1())
        return [
            id,
            metadata,
            node.get_content(metadata_mode=MetadataMode.NONE),
            node.embedding,
        ]

    def add(self, embedding_results: List[BaseNode]) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in embedding_results]
        ids = [result[0] for result in rows_to_insert]
        self._sync_client.upsert(rows_to_insert)
        return ids

    async def async_add(self, embedding_results: List[BaseNode]) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in embedding_results]
        ids = [result.node_id for result in embedding_results]
        await self._async_client.upsert(rows_to_insert)
        return ids

    def _filter_to_dict(
        self, metadata_filters: Optional[MetadataFilters]
    ) -> Optional[Dict[str, str]]:
        if metadata_filters is None:
            return None

        res = {}
        for filter in metadata_filters.filters:
            res[filter.key] = filter.value

        return res

    def _db_rows_to_query_result(self, rows: List) -> VectorStoreQueryResult:
        from timescale_vector import client

        nodes = []
        similarities = []
        ids = []
        for row in rows:
            try:
                node = metadata_dict_to_node(row[client.SEARCH_RESULT_METADATA_IDX])
                node.set_content(str(row[client.SEARCH_RESULT_CONTENTS_IDX]))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                node = TextNode(
                    id_=row[client.SEARCH_RESULT_ID_IDX],
                    text=row[client.SEARCH_RESULT_CONTENTS_IDX],
                    metadata=row[client.SEARCH_RESULT_METADATA_IDX],
                )
            similarities.append(row[client.SEARCH_RESULT_DISTANCE_IDX])
            ids.append(row[client.SEARCH_RESULT_ID_IDX])
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def date_to_range_filter(self, **kwargs: Any) -> Any:
        constructor_args = {
            key: kwargs[key]
            for key in [
                "start_date",
                "end_date",
                "time_delta",
                "start_inclusive",
                "end_inclusive",
            ]
            if key in kwargs
        }
        if not constructor_args or len(constructor_args) == 0:
            return None

        try:
            from timescale_vector import client
        except ImportError:
            raise ValueError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )
        return client.UUIDTimeRange(**constructor_args)

    def _query_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        filter = self._filter_to_dict(metadata_filters)
        res = self._sync_client.search(
            embedding,
            limit,
            filter,
            uuid_time_filter=self.date_to_range_filter(**kwargs),
        )
        return self._db_rows_to_query_result(res)

    async def _aquery_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        filter = self._filter_to_dict(metadata_filters)
        res = await self._async_client.search(
            embedding,
            limit,
            filter,
            uuid_time_filter=self.date_to_range_filter(**kwargs),
        )
        return self._db_rows_to_query_result(res)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return self._query_with_score(
            query.query_embedding, query.similarity_top_k, query.filters, **kwargs
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return await self._aquery_with_score(
            query.query_embedding,
            query.similarity_top_k,
            query.filters,
            **kwargs,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        filter: Dict[str, str] = {"doc_id": ref_doc_id}
        self._sync_client.delete_by_metadata(filter)

    DEFAULT_INDEX_TYPE = IndexType.TIMESCALE_VECTOR

    def create_index(
        self, index_type: IndexType = DEFAULT_INDEX_TYPE, **kwargs: Any
    ) -> None:
        try:
            from timescale_vector import client
        except ImportError:
            raise ValueError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )

        if index_type == IndexType.PGVECTOR_IVFFLAT:
            self._sync_client.create_embedding_index(client.IvfflatIndex(**kwargs))

        if index_type == IndexType.PGVECTOR_HNSW:
            self._sync_client.create_embedding_index(client.HNSWIndex(**kwargs))

        if index_type == IndexType.TIMESCALE_VECTOR:
            self._sync_client.create_embedding_index(
                client.TimescaleVectorIndex(**kwargs)
            )

    def drop_index(self) -> None:
        self._sync_client.drop_embedding_index()
