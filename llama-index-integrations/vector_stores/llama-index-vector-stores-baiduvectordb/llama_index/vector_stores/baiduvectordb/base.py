"""A store that is built with Baidu VectorDB."""

import json
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import DEFAULT_DOC_ID_KEY, DEFAULT_TEXT_KEY

logger = logging.getLogger(__name__)

DEFAULT_ACCOUNT = "root"
DEFAULT_DATABASE_NAME = "llama_default_database"
DEFAULT_TABLE_NAME = "llama_default_table"
DEFAULT_TIMEOUT_IN_MILLS: int = 30 * 1000
DEFAULT_WAIT_TIMEOUT: int = 30  # 30 seconds for operations

DEFAULT_PARTITION = 1
DEFAULT_REPLICA = 3
DEFAULT_INDEX_TYPE = "HNSW"
DEFAULT_METRIC_TYPE = "L2"

DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF = 10

FIELD_ID: str = "id"
FIELD_VECTOR: str = "vector"
FIELD_METADATA: str = "metadata"
INDEX_SUFFIX: str = "_index"
INDEX_VECTOR: str = "vector_index"

VALUE_NONE_ERROR = "Parameter `{}` can not be None."
VALUE_RANGE_ERROR = "The value of parameter `{}` must be within {}."
NOT_SUPPORT_INDEX_TYPE_ERROR = (
    "Unsupported index type: `{}`, supported index types are {}"
)
NOT_SUPPORT_METRIC_TYPE_ERROR = (
    "Unsupported metric type: `{}`, supported metric types are {}"
)


def _try_import() -> None:
    try:
        import pymochow  # noqa: F401
    except ImportError:
        raise ImportError(
            "`pymochow` package not found, please run `pip install pymochow`"
        )


class TableField:
    name: str
    data_type: str = "STRING"

    def __init__(self, name: str, data_type: str = "STRING"):
        self.name = name
        self.data_type = "STRING" if data_type is None else data_type


class TableParams:
    """
    Baidu VectorDB table params.

    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/s/mlrsob0p6

    Args:
        dimension int: The dimension of vector.
        replication int: The number of replicas in the table.
        partition int: The number of partitions in the table.
        index_type (Optional[str]): HNSW, FLAT... Default value is "HNSW"
        metric_type (Optional[str]): L2, COSINE, IP. Default value is "L2"
        drop_exists (Optional[bool]): Delete the existing Table. Default value is False.
        vector_params (Optional[Dict]):
          if HNSW set parameters: `M` and `efConstruction`, for example `{'M': 16, efConstruction: 200}`
          default is HNSW
        filter_fields: Optional[List[str]]: Set the fields for filtering, The
        fields used for filtering must have a value in every row of the table
        and cannot be null.
          for example: ['author', 'age']
          This can be used when calling the query method：
             store.add([
                TextNode(..., metadata={'age'=23, 'name'='name1'})
            ])
             ...
             query = VectorStoreQuery(...)
             store.query(query, filter="age > 20 and age < 40 and name = 'name1'")

    """

    def __init__(
        self,
        dimension: int,
        table_name: str = DEFAULT_TABLE_NAME,
        replication: int = DEFAULT_REPLICA,
        partition: int = DEFAULT_PARTITION,
        index_type: str = DEFAULT_INDEX_TYPE,
        metric_type: str = DEFAULT_METRIC_TYPE,
        drop_exists: Optional[bool] = False,
        vector_params: Optional[Dict] = None,
        filter_fields: Optional[List[TableField]] = None,
    ):
        if filter_fields is None:
            filter_fields = []
        self.dimension = dimension
        self.table_name = table_name
        self.replication = replication
        self.partition = partition
        self.index_type = index_type
        self.metric_type = metric_type
        self.drop_exists = drop_exists
        self.vector_params = vector_params
        self.filter_fields = filter_fields


class BaiduVectorDB(BasePydanticVectorStore):
    """
    Baidu VectorDB as a vector store.

    In order to use this you need to have a database instance.
    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/index.html

    Args:
        endpoint (Optional[str]): endpoint of Baidu VectorDB
        account (Optional[str]): The account for Baidu VectorDB. Default value is "root"
        api_key (Optional[str]): The Api-Key for Baidu VectorDB
        database_name(Optional[str]): The database name for Baidu VectorDB
        table_params (Optional[TableParams]): The table parameters for BaiduVectorDB

    """

    user_defined_fields: List[TableField] = Field(default_factory=list)
    batch_size: int

    _vdb_client: Any = PrivateAttr()
    _database: Any = PrivateAttr()
    _table: Any = PrivateAttr()

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        account: str = DEFAULT_ACCOUNT,
        database_name: str = DEFAULT_DATABASE_NAME,
        table_params: TableParams = TableParams(dimension=1536),
        batch_size: int = 1000,
        stores_text: bool = True,
        **kwargs: Any,
    ):
        """Init params."""
        super().__init__(
            user_defined_fields=table_params.filter_fields,
            batch_size=batch_size,
            stores_text=stores_text,
            **kwargs,
        )

        self._init_client(endpoint, account, api_key)
        self._create_database_if_not_exists(database_name)
        self._create_table(table_params)

    @classmethod
    def class_name(cls) -> str:
        return "BaiduVectorDB"

    @classmethod
    def from_params(
        cls,
        endpoint: str,
        api_key: str,
        account: str = DEFAULT_ACCOUNT,
        database_name: str = DEFAULT_DATABASE_NAME,
        table_params: TableParams = TableParams(dimension=1536),
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> "BaiduVectorDB":
        _try_import()
        return cls(
            endpoint=endpoint,
            account=account,
            api_key=api_key,
            database_name=database_name,
            table_params=table_params,
            batch_size=batch_size,
            **kwargs,
        )

    def _init_client(self, endpoint: str, account: str, api_key: str) -> None:
        import pymochow
        from pymochow.configuration import Configuration
        from pymochow.auth.bce_credentials import BceCredentials

        logger.debug("Connecting to Baidu VectorDB...")
        config = Configuration(
            credentials=BceCredentials(account, api_key),
            endpoint=endpoint,
            connection_timeout_in_mills=DEFAULT_TIMEOUT_IN_MILLS,
        )
        self._vdb_client = pymochow.MochowClient(config)
        logger.debug("Baidu VectorDB client initialized.")

    def _create_database_if_not_exists(self, database_name: str) -> None:
        db_list = self._vdb_client.list_databases()

        if database_name in [db.database_name for db in db_list]:
            logger.debug(f"Database '{database_name}' already exists.")
            self._database = self._vdb_client.database(database_name)
        else:
            logger.debug(f"Creating database '{database_name}'.")
            self._database = self._vdb_client.create_database(database_name)
            logger.debug(f"Database '{database_name}' created.")

    def _create_table(self, table_params: TableParams) -> None:
        import pymochow

        if table_params is None:
            raise ValueError(VALUE_NONE_ERROR.format("table_params"))

        try:
            self._table = self._database.describe_table(table_params.table_name)
            logger.debug(f"Table '{table_params.table_name}' already exists.")
            if table_params.drop_exists:
                logger.debug(f"Dropping table '{table_params.table_name}'.")
                self._database.drop_table(table_params.table_name)
                # wait for table to be fully dropped
                start_time = time.time()
                loop_count = 0
                while time.time() - start_time < DEFAULT_WAIT_TIMEOUT:
                    loop_count += 1
                    logger.debug(
                        f"Waiting for table {table_params.table_name} to be dropped,"
                        f" attempt {loop_count}"
                    )
                    time.sleep(1)
                    tables = self._database.list_table()
                    table_names = {table.table_name for table in tables}
                    if table_params.table_name not in table_names:
                        logger.debug(f"Table '{table_params.table_name}' dropped.")
                        break
                else:
                    raise TimeoutError(
                        f"Table {table_params.table_name} was not dropped within"
                        f" {DEFAULT_WAIT_TIMEOUT} seconds"
                    )
                self._create_table_in_db(table_params)
        except pymochow.exception.ServerError:
            self._create_table_in_db(table_params)

    def _create_table_in_db(
        self,
        table_params: TableParams,
    ) -> None:
        from pymochow.model.enum import FieldType, TableState
        from pymochow.model.schema import Field, Schema, SecondaryIndex, VectorIndex
        from pymochow.model.table import Partition

        logger.debug(f"Creating table '{table_params.table_name}'.")
        index_type = self._get_index_type(table_params.index_type)
        metric_type = self._get_metric_type(table_params.metric_type)
        vector_params = self._get_index_params(index_type, table_params)
        fields = []
        fields.append(
            Field(
                FIELD_ID,
                FieldType.STRING,
                primary_key=True,
                partition_key=True,
                auto_increment=False,
                not_null=True,
            )
        )
        fields.append(Field(DEFAULT_DOC_ID_KEY, FieldType.STRING))
        fields.append(Field(FIELD_METADATA, FieldType.STRING))
        fields.append(Field(DEFAULT_TEXT_KEY, FieldType.STRING))
        fields.append(
            Field(
                FIELD_VECTOR, FieldType.FLOAT_VECTOR, dimension=table_params.dimension
            )
        )
        for field in table_params.filter_fields:
            fields.append(Field(field.name, FieldType(field.data_type), not_null=True))

        indexes = []
        indexes.append(
            VectorIndex(
                index_name=INDEX_VECTOR,
                index_type=index_type,
                field=FIELD_VECTOR,
                metric_type=metric_type,
                params=vector_params,
            )
        )
        for field in table_params.filter_fields:
            index_name = field.name + INDEX_SUFFIX
            indexes.append(SecondaryIndex(index_name=index_name, field=field.name))

        schema = Schema(fields=fields, indexes=indexes)
        self._table = self._database.create_table(
            table_name=table_params.table_name,
            replication=table_params.replication,
            partition=Partition(partition_num=table_params.partition),
            schema=schema,
            enable_dynamic_field=True,
        )
        # wait for table to be ready
        start_time = time.time()
        loop_count = 0
        while time.time() - start_time < DEFAULT_WAIT_TIMEOUT:
            loop_count += 1
            logger.debug(
                f"Waiting for table {table_params.table_name} to become ready,"
                f" attempt {loop_count}"
            )
            time.sleep(1)
            table = self._database.describe_table(table_params.table_name)
            if table.state == TableState.NORMAL:
                logger.debug(f"Table '{table_params.table_name}' is ready.")
                break
        else:
            raise TimeoutError(
                f"Table {table_params.table_name} did not become ready within"
                f" {DEFAULT_WAIT_TIMEOUT} seconds"
            )

    @staticmethod
    def _get_index_params(index_type: Any, table_params: TableParams) -> None:
        from pymochow.model.enum import IndexType
        from pymochow.model.schema import HNSWParams

        vector_params = (
            {} if table_params.vector_params is None else table_params.vector_params
        )

        if index_type == IndexType.HNSW:
            return HNSWParams(
                m=vector_params.get("M", DEFAULT_HNSW_M),
                efconstruction=vector_params.get(
                    "efConstruction", DEFAULT_HNSW_EF_CONSTRUCTION
                ),
            )
        return None

    @staticmethod
    def _get_index_type(index_type_value: str) -> Any:
        from pymochow.model.enum import IndexType

        index_type_value = index_type_value or IndexType.HNSW
        try:
            return IndexType(index_type_value)
        except ValueError:
            support_index_types = [d.value for d in IndexType.__members__.values()]
            raise ValueError(
                NOT_SUPPORT_INDEX_TYPE_ERROR.format(
                    index_type_value, support_index_types
                )
            )

    @staticmethod
    def _get_metric_type(metric_type_value: str) -> Any:
        from pymochow.model.enum import MetricType

        metric_type_value = metric_type_value or MetricType.L2
        try:
            return MetricType(metric_type_value.upper())
        except ValueError:
            support_metric_types = [d.value for d in MetricType.__members__.values()]
            raise ValueError(
                NOT_SUPPORT_METRIC_TYPE_ERROR.format(
                    metric_type_value, support_metric_types
                )
            )

    @property
    def client(self) -> Any:
        """Get client."""
        return self._vdb_client

    def clear(self) -> None:
        """
        Clear all nodes from Baidu VectorDB table.
        This method deletes the table.
        """
        return asyncio.get_event_loop().run_until_complete(self.aclear())

    async def aclear(self) -> None:
        """
        Asynchronously clear all nodes from Baidu VectorDB table.
        This method deletes the table.
        """
        import pymochow

        try:
            # Check if table exists
            table_name = self._table.table_name
            self._database.describe_table(table_name)
            # Table exists, drop it
            logger.debug(f"Dropping table '{table_name}'.")
            self._database.drop_table(table_name)
            # Wait for table to be fully dropped
            start_time = time.time()
            loop_count = 0
            while time.time() - start_time < DEFAULT_WAIT_TIMEOUT:
                loop_count += 1
                logger.debug(
                    f"Waiting for table {table_name} to be dropped, attempt {loop_count}"
                )
                await asyncio.sleep(1)
                tables = self._database.list_table()
                table_names = {table.table_name for table in tables}
                if table_name not in table_names:
                    logger.debug(f"Table '{table_name}' dropped.")
                    break
            else:
                raise TimeoutError(
                    f"Table {table_name} was not dropped within {DEFAULT_WAIT_TIMEOUT}"
                    " seconds"
                )
        except (pymochow.exception.ServerError, AttributeError):
            # Table doesn't exist or _table not properly initialized, nothing to delete
            logger.debug("Table does not exist, nothing to clear.")

    def add(
        self,
        nodes: List[BaseNode],
        *,
        rebuild_index: bool = True,
        rebuild_timeout: Optional[int] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to Baidu VectorDB table.

        Args:
            nodes: List of nodes with embeddings.
            rebuild_index: Optional. Whether to rebuild the vector index
                          after adding nodes. Defaults to True.
            rebuild_timeout: Optional. Timeout for rebuilding the index in seconds.
                             If None, it will wait indefinitely. Defaults to None.

        Returns:
            List of node IDs that were added to the table.

        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(
                nodes,
                rebuild_index=rebuild_index,
                rebuild_timeout=rebuild_timeout,
                **add_kwargs,
            )
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        *,
        rebuild_index: bool = True,
        rebuild_timeout: Optional[int] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Asynchronous method to add nodes to Baidu VectorDB table.

        Args:
            nodes: List of nodes with embeddings.
            rebuild_index: Optional. Whether to rebuild the vector index
                          after adding nodes. Defaults to True.
            rebuild_timeout: Optional. Timeout for rebuilding the index in seconds.
                             If None, it will wait indefinitely. Defaults to None.

        Returns:
            List of node IDs that were added to the table.

        """
        if len(nodes) == 0:
            return []

        from pymochow.model.table import Row
        from pymochow.model.enum import IndexState

        ids = []
        rows = []
        for i, node in enumerate(nodes):
            logger.debug(f"Processing node {i + 1}/{len(nodes)}, id: {node.node_id}")
            row = Row(id=node.node_id, vector=node.get_embedding())
            if node.ref_doc_id is not None:
                row._data[DEFAULT_DOC_ID_KEY] = node.ref_doc_id
            if node.metadata is not None:
                row._data[FIELD_METADATA] = json.dumps(node.metadata)
                for field in self.user_defined_fields:
                    v = node.metadata.get(field.name)
                    if v is not None:
                        row._data[field.name] = v
            if isinstance(node, TextNode) and node.text is not None:
                row._data[DEFAULT_TEXT_KEY] = node.text

            rows.append(row)
            ids.append(node.node_id)

            if len(rows) >= self.batch_size:
                logger.debug(f"Upserting {len(rows)} rows to the table.")
                self._table.upsert(rows=rows)
                rows = []

        if len(rows) > 0:
            logger.debug(f"Upserting remaining {len(rows)} rows to the table.")
            self._table.upsert(rows=rows)

        if rebuild_index:
            logger.debug(f"Rebuilding index '{INDEX_VECTOR}'.")
            self._table.rebuild_index(INDEX_VECTOR)
            start_time = time.time()
            loop_count = 0
            while True:
                loop_count += 1
                logger.debug(
                    f"Waiting for index {INDEX_VECTOR} to be ready, attempt"
                    f" {loop_count}"
                )
                await asyncio.sleep(1)
                index = self._table.describe_index(INDEX_VECTOR)
                if index.state == IndexState.NORMAL:
                    logger.debug(f"Index '{INDEX_VECTOR}' is ready.")
                    break
                if (
                    rebuild_timeout is not None
                    and time.time() - start_time > rebuild_timeout
                ):
                    raise TimeoutError(
                        f"Index {INDEX_VECTOR} did not become ready within"
                        f" {rebuild_timeout} seconds"
                    )

        return ids

    # Baidu VectorDB Not support delete with filter right now, will support it later.
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id or ids.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Not support.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): contains
                query_embedding (List[float]): query embedding
                similarity_top_k (int): top k most similar nodes
                filters (Optional[MetadataFilters]): filter result

        Returns:
            VectorStoreQueryResult: Query result containing nodes, similarities, and ids.

        """
        return asyncio.get_event_loop().run_until_complete(self.aquery(query, **kwargs))

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): contains
                query_embedding (List[float]): query embedding
                similarity_top_k (int): top k most similar nodes
                filters (Optional[MetadataFilters]): filter result

        Returns:
            VectorStoreQueryResult: Query result containing nodes, similarities, and ids.

        """
        from pymochow.model.table import AnnSearch, HNSWSearchParams

        search_filter = None
        if query.filters is not None:
            search_filter = self._build_filter_condition(query.filters, **kwargs)
        logger.debug(
            f"Querying with top_k={query.similarity_top_k} and filter='{search_filter}'"
        )
        anns = AnnSearch(
            vector_field=FIELD_VECTOR,
            vector_floats=query.query_embedding,
            params=HNSWSearchParams(ef=DEFAULT_HNSW_EF, limit=query.similarity_top_k),
            filter=search_filter,
        )
        res = self._table.search(anns=anns, retrieve_vector=True)
        rows = res.rows
        if rows is None or len(rows) == 0:
            logger.debug("Query returned no results.")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        logger.debug(f"Query returned {len(rows)} results.")
        nodes = []
        similarities = []
        ids = []
        for row in rows:
            similarities.append(row.get("distance"))
            row_data = row.get("row", {})
            ids.append(row_data.get(FIELD_ID))

            meta_str = row_data.get(FIELD_METADATA)
            meta = {} if meta_str is None else json.loads(meta_str)
            doc_id = row_data.get(DEFAULT_DOC_ID_KEY)

            node = TextNode(
                id_=row_data.get(FIELD_ID),
                text=row_data.get(DEFAULT_TEXT_KEY),
                embedding=row_data.get(FIELD_VECTOR),
                metadata=meta,
            )
            if doc_id is not None:
                node.relationships = {
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                }

            nodes.append(node)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @staticmethod
    def _build_filter_condition(standard_filters: MetadataFilters) -> str:
        filters_list = []

        for filter in standard_filters.filters:
            value = (
                f"'{filter.value}'"
                if isinstance(filter.value, (str, bool))
                else filter.value
            )

            if filter.operator:
                if filter.operator.value in ["<", ">", "<=", ">=", "!="]:
                    condition = f"{filter.key} {filter.operator.value} {value}"
                elif filter.operator.value in ["=="]:
                    condition = f"{filter.key} == {value}"
                else:
                    raise ValueError(
                        f"Filter operator {filter.operator} not supported."
                    )
            else:
                condition = f"{filter.key} = {value}"

            filters_list.append(condition)

        return f" {standard_filters.condition.value.upper()} ".join(filters_list)
