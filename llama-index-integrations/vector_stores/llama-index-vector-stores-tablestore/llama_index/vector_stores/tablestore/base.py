"""Tablestore vector store."""

import json
import traceback
from logging import getLogger
from typing import Any, List, Optional, Dict

import tablestore
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterCondition,
    MetadataFilter,
    FilterOperator,
)


class TablestoreVectorStore(BasePydanticVectorStore):
    """`Tablestore` vector store.

    To use, you should have the ``tablestore`` python package installed.

    Examples:
        ```python
        import tablestore
        import os

        store = TablestoreVectorStore(
            endpoint=os.getenv("end_point"),
            instance_name=os.getenv("instance_name"),
            access_key_id=os.getenv("access_key_id"),
            access_key_secret=os.getenv("access_key_secret"),
            vector_dimension=512,
            vector_metric_type=tablestore.VectorMetricType.VM_COSINE,
            # metadata mapping is used to filter non-vector fields.
            metadata_mappings=[
                tablestore.FieldSchema(
                    "type",
                    tablestore.FieldType.KEYWORD,
                    index=True,
                    enable_sort_and_agg=True,
                ),
                tablestore.FieldSchema(
                    "time", tablestore.FieldType.LONG, index=True, enable_sort_and_agg=True
                ),
            ],
        )
        ```
    """

    is_embedding_query: bool = True
    stores_text: bool = True
    _logger: Any = PrivateAttr(default=None)
    _tablestore_client: tablestore.OTSClient = PrivateAttr(default=None)
    _table_name: str = PrivateAttr(default="llama_index_vector_store_ots_v1")
    _index_name: str = PrivateAttr(default="llama_index_vector_store_ots_index_v1")
    _text_field: str = PrivateAttr(default="content")
    _vector_field: str = PrivateAttr(default="embedding")
    _metadata_mappings: List[tablestore.FieldSchema] = PrivateAttr(default=None)

    def __init__(
        self,
        tablestore_client: Optional[tablestore.OTSClient] = None,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        table_name: str = "llama_index_vector_store_ots_v1",
        index_name: str = "llama_index_vector_store_ots_index_v1",
        text_field: str = "content",
        vector_field: str = "embedding",
        vector_dimension: int = 512,
        vector_metric_type: tablestore.VectorMetricType = tablestore.VectorMetricType.VM_COSINE,
        metadata_mappings: Optional[List[tablestore.FieldSchema]] = None,
    ) -> None:
        super().__init__()
        self._logger = getLogger(__name__)
        if not tablestore_client:
            self._tablestore_client = tablestore.OTSClient(
                endpoint,
                access_key_id,
                access_key_secret,
                instance_name,
                retry_policy=tablestore.WriteRetryPolicy(),
            )
        else:
            self._tablestore_client = tablestore_client
        self._table_name = table_name
        self._index_name = index_name
        self._text_field = text_field
        self._vector_field = vector_field

        self._metadata_mappings = [
            tablestore.FieldSchema(
                text_field,
                tablestore.FieldType.TEXT,
                index=True,
                enable_sort_and_agg=False,
                store=False,
                analyzer=tablestore.AnalyzerType.MAXWORD,
            ),
            tablestore.FieldSchema(
                vector_field,
                tablestore.FieldType.VECTOR,
                vector_options=tablestore.VectorOptions(
                    data_type=tablestore.VectorDataType.VD_FLOAT_32,
                    dimension=vector_dimension,
                    metric_type=vector_metric_type,
                ),
            ),
        ]
        if metadata_mappings:
            for mapping in metadata_mappings:
                if (
                    mapping.field_name == text_field
                    or mapping.field_name == vector_field
                ):
                    continue
                self._metadata_mappings.append(mapping)

    def create_table_if_not_exist(self) -> None:
        """Create table if not exist."""
        table_list = self._tablestore_client.list_table()
        if self._table_name in table_list:
            self._logger.info(
                "Tablestore system table[%s] already exists", self._table_name
            )
            return
        self._logger.info(
            "Tablestore system table[%s] does not exist, try to create the table.",
            self._table_name,
        )

        schema_of_primary_key = [("id", "STRING")]
        table_meta = tablestore.TableMeta(self._table_name, schema_of_primary_key)
        table_options = tablestore.TableOptions()
        reserved_throughput = tablestore.ReservedThroughput(
            tablestore.CapacityUnit(0, 0)
        )
        try:
            self._tablestore_client.create_table(
                table_meta, table_options, reserved_throughput
            )
            self._logger.info(
                "Tablestore create table[%s] successfully.", self._table_name
            )
        except tablestore.OTSClientError as e:
            traceback.print_exc()
            self._logger.exception(
                "Tablestore create system table[%s] failed with client error, http_status:%d, error_message:%s",
                self._table_name,
                e.get_http_status(),
                e.get_error_message(),
            )
        except tablestore.OTSServiceError as e:
            traceback.print_exc()
            self._logger.exception(
                "Tablestore create system table[%s] failed with client error, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                self._table_name,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def create_search_index_if_not_exist(self) -> None:
        """Create search index if not exist."""
        search_index_list = self._tablestore_client.list_search_index(
            table_name=self._table_name
        )
        if self._index_name in [t[1] for t in search_index_list]:
            self._logger.info(
                "Tablestore system index[%s] already exists", self._index_name
            )
            return
        index_meta = tablestore.SearchIndexMeta(self._metadata_mappings)
        self._tablestore_client.create_search_index(
            self._table_name, self._index_name, index_meta
        )
        self._logger.info(
            "Tablestore create system index[%s] successfully.", self._index_name
        )

    def delete_table_if_exists(self):
        """Delete table if exists."""
        search_index_list = self._tablestore_client.list_search_index(
            table_name=self._table_name
        )
        for resp_tuple in search_index_list:
            self._tablestore_client.delete_search_index(resp_tuple[0], resp_tuple[1])
            self._logger.info(
                "Tablestore delete index[%s] successfully.", self._index_name
            )
        self._tablestore_client.delete_table(self._table_name)
        self._logger.info(
            "Tablestore delete system table[%s] successfully.", self._index_name
        )

    def delete_search_index(self, table_name, index_name) -> None:
        self._tablestore_client.delete_search_index(table_name, index_name)
        self._logger.info("Tablestore delete index[%s] successfully.", self._index_name)

    def _write_row(
        self,
        row_id: str,
        content: str,
        embedding_vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        primary_key = [("id", row_id)]
        attribute_columns = [
            (self._text_field, content),
            (self._vector_field, json.dumps(embedding_vector)),
        ]
        for k, v in metadata.items():
            item = (k, v)
            attribute_columns.append(item)
        row = tablestore.Row(primary_key, attribute_columns)

        try:
            self._tablestore_client.put_row(self._table_name, row)
            self._logger.debug(
                "Tablestore put row successfully. id:%s, content:%s, meta_data:%s",
                row_id,
                content,
                metadata,
            )
        except tablestore.OTSClientError as e:
            self._logger.exception(
                "Tablestore put row failed with client error:%s, id:%s, content:%s, meta_data:%s",
                e,
                row_id,
                content,
                metadata,
            )
        except tablestore.OTSServiceError as e:
            self._logger.exception(
                "Tablestore put row failed with client error:%s, id:%s, content:%s, meta_data:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                row_id,
                content,
                metadata,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def _delete_row(self, row_id: str) -> None:
        primary_key = [("id", row_id)]
        row = tablestore.Row(primary_key)
        try:
            self._tablestore_client.delete_row(self._table_name, row, None)
            self._logger.info("Tablestore delete row successfully. id:%s", row_id)
        except tablestore.OTSClientError as e:
            self._logger.exception(
                "Tablestore delete row failed with client error:%s, id:%s", e, row_id
            )
        except tablestore.OTSServiceError as e:
            self._logger.exception(
                "Tablestore delete row failed with client error:%s, id:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                row_id,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def _delete_all(self) -> None:
        inclusive_start_primary_key = [("id", tablestore.INF_MIN)]
        exclusive_end_primary_key = [("id", tablestore.INF_MAX)]
        total = 0
        try:
            while True:
                (
                    consumed,
                    next_start_primary_key,
                    row_list,
                    next_token,
                ) = self._tablestore_client.get_range(
                    self._table_name,
                    tablestore.Direction.FORWARD,
                    inclusive_start_primary_key,
                    exclusive_end_primary_key,
                    [],
                    5000,
                    max_version=1,
                )
                for row in row_list:
                    self._tablestore_client.delete_row(self._table_name, row, None)
                    total += 1
                if next_start_primary_key is not None:
                    inclusive_start_primary_key = next_start_primary_key
                else:
                    break
        except tablestore.OTSClientError as e:
            self._logger.exception(
                "Tablestore delete row failed with client error:%s", e
            )
        except tablestore.OTSServiceError as e:
            self._logger.exception(
                "Tablestore delete row failed with client error:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )
        self._logger.info("delete all rows count:%d", total)

    def _search(
        self, query: VectorStoreQuery, knn_top_k: int
    ) -> VectorStoreQueryResult:
        filter_query = self._parse_filters(query.filters)
        ots_query = tablestore.KnnVectorQuery(
            field_name=self._vector_field,
            top_k=knn_top_k,
            float32_query_vector=query.query_embedding,
            filter=filter_query,
        )
        sort = tablestore.Sort(
            sorters=[tablestore.ScoreSort(sort_order=tablestore.SortOrder.DESC)]
        )
        search_query = tablestore.SearchQuery(
            ots_query, limit=query.similarity_top_k, get_total_count=False, sort=sort
        )
        try:
            search_response = self._tablestore_client.search(
                table_name=self._table_name,
                index_name=self._index_name,
                search_query=search_query,
                columns_to_get=tablestore.ColumnsToGet(
                    return_type=tablestore.ColumnReturnType.ALL
                ),
            )
            self._logger.info(
                "Tablestore search successfully. request_id:%s",
                search_response.request_id,
            )
            return self._to_query_result(search_response)
        except tablestore.OTSClientError as e:
            self._logger.exception("Tablestore search failed with client error:%s", e)
        except tablestore.OTSServiceError as e:
            self._logger.exception(
                "Tablestore search failed with client error:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def _to_query_result(self, search_response) -> VectorStoreQueryResult:
        nodes = []
        ids = []
        similarities = []
        for hit in search_response.search_hits:
            row = hit.row
            score = hit.score
            node_id = row[0][0][1]
            meta_data = {}
            text = None
            embedding = None
            for col in row[1]:
                key = col[0]
                val = col[1]
                if key == self._text_field:
                    text = val
                    continue
                if key == self._vector_field:
                    embedding = json.loads(val)
                    continue
                meta_data[key] = val
            node = TextNode(
                id_=node_id,
                text=text,
                metadata=meta_data,
                embedding=embedding,
            )
            ids.append(node_id)
            nodes.append(node)
            similarities.append(score)
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=similarities)

    def _parse_filters_recursively(
        self, filters: MetadataFilters
    ) -> tablestore.BoolQuery:
        """Parse (possibly nested) MetadataFilters to equivalent tablestore search expression."""
        bool_query = tablestore.BoolQuery()
        if filters.condition is FilterCondition.AND:
            bool_clause = bool_query.must_queries
        elif filters.condition is FilterCondition.OR:
            bool_clause = bool_query.should_queries
        else:
            raise ValueError(f"Unsupported filter condition: {filters.condition}")

        for filter_item in filters.filters:
            if isinstance(filter_item, MetadataFilter):
                bool_clause.append(self._parse_filter(filter_item))
            elif isinstance(filter_item, MetadataFilters):
                bool_clause.append(self._parse_filters_recursively(filter_item))
            else:
                raise ValueError(f"Unsupported filter type: {type(filter_item)}")

        return bool_query

    def _parse_filters(self, filters: Optional[MetadataFilters]) -> tablestore.Query:
        """Parse MetadataFilters to equivalent OpenSearch expression."""
        if filters is None:
            return tablestore.MatchAllQuery()
        return self._parse_filters_recursively(filters=filters)

    @staticmethod
    def _parse_filter(filter_item: MetadataFilter) -> tablestore.Query:
        key = filter_item.key
        val = filter_item.value
        op = filter_item.operator

        if op == FilterOperator.EQ:
            return tablestore.TermQuery(field_name=key, column_value=val)
        elif op == FilterOperator.GT:
            return tablestore.RangeQuery(
                field_name=key, range_from=val, include_lower=False
            )
        elif op == FilterOperator.GTE:
            return tablestore.RangeQuery(
                field_name=key, range_from=val, include_lower=True
            )
        elif op == FilterOperator.LT:
            return tablestore.RangeQuery(
                field_name=key, range_to=val, include_upper=False
            )
        elif op == FilterOperator.LTE:
            return tablestore.RangeQuery(
                field_name=key, range_to=val, include_upper=True
            )
        elif op == FilterOperator.NE:
            bq = tablestore.BoolQuery()
            bq.must_not_queries.append(
                tablestore.TermQuery(field_name=key, column_value=val)
            )
            return bq
        elif op in [FilterOperator.IN, FilterOperator.ANY]:
            return tablestore.TermsQuery(field_name=key, column_values=val)
        elif op == FilterOperator.NIN:
            bq = tablestore.BoolQuery()
            bq.must_not_queries.append(
                tablestore.TermsQuery(field_name=key, column_values=val)
            )
            return bq
        elif op == FilterOperator.ALL:
            bq = tablestore.BoolQuery()
            for val_item in val:
                bq.must_queries.append(
                    tablestore.TermQuery(field_name=key, column_value=val_item)
                )
            return bq
        elif op == FilterOperator.TEXT_MATCH:
            return tablestore.MatchQuery(field_name=key, text=val)
        elif op == FilterOperator.CONTAINS:
            return tablestore.WildcardQuery(field_name=key, value=f"*{val}*")
        else:
            raise ValueError(f"Unsupported filter operator: {filter_item.operator}")

    @property
    def client(self) -> Any:
        """Get client."""
        return self._tablestore_client

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Add nodes to vector store."""
        if len(nodes) == 0:
            return []
        ids = []
        for node in nodes:
            self._write_row(
                row_id=node.node_id,
                content=node.text,
                embedding_vector=node.get_embedding(),
                metadata=node.metadata,
            )
            ids.append(node.node_id)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id."""
        self._delete_row(ref_doc_id)

    def clear(self) -> None:
        """Clear all nodes from configured vector store."""
        self._delete_all()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        knn_top_k = query.similarity_top_k
        if "knn_top_k" in kwargs:
            knn_top_k = kwargs["knn_top_k"]
        return self._search(query=query, knn_top_k=knn_top_k)
