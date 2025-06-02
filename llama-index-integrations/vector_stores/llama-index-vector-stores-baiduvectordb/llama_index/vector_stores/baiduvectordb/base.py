"""A store that is built with Baidu VectorDB."""

import json
import time
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

DEFAULT_ACCOUNT = "root"
DEFAULT_DATABASE_NAME = "llama_default_database"
DEFAULT_TABLE_NAME = "llama_default_table"
DEFAULT_TIMEOUT_IN_MILLS: int = 30 * 1000

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
        **kwargs: Any,
    ):
        """Init params."""
        super().__init__(
            user_defined_fields=table_params.filter_fields,
            batch_size=batch_size,
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

        config = Configuration(
            credentials=BceCredentials(account, api_key),
            endpoint=endpoint,
            connection_timeout_in_mills=DEFAULT_TIMEOUT_IN_MILLS,
        )
        self._vdb_client = pymochow.MochowClient(config)

    def _create_database_if_not_exists(self, database_name: str) -> None:
        db_list = self._vdb_client.list_databases()

        if database_name in [db.database_name for db in db_list]:
            self._database = self._vdb_client.database(database_name)
        else:
            self._database = self._vdb_client.create_database(database_name)

    def _create_table(self, table_params: TableParams) -> None:
        import pymochow

        if table_params is None:
            raise ValueError(VALUE_NONE_ERROR.format("table_params"))

        try:
            self._table = self._database.describe_table(table_params.table_name)
            if table_params.drop_exists:
                self._database.drop_table(table_params.table_name)
                # wait db release resource
                time.sleep(5)
                self._create_table_in_db(table_params)
        except pymochow.exception.ServerError:
            self._create_table_in_db(table_params)

    def _create_table_in_db(
        self,
        table_params: TableParams,
    ) -> None:
        from pymochow.model.enum import FieldType
        from pymochow.model.schema import Field, Schema, SecondaryIndex, VectorIndex
        from pymochow.model.table import Partition

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
            schema=Schema(fields=fields, indexes=indexes),
            enable_dynamic_field=True,
        )
        # need wait 10s to wait proxy sync meta
        time.sleep(10)

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
        return self.tencent_client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        from pymochow.model.table import Row
        from pymochow.model.enum import IndexState

        ids = []
        rows = []
        for node in nodes:
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
                self.collection.upsert(rows=rows)
                rows = []

        if len(rows) > 0:
            self._table.upsert(rows=rows)

        self._table.rebuild_index(INDEX_VECTOR)
        while True:
            time.sleep(2)
            index = self._table.describe_index(INDEX_VECTOR)
            if index.state == IndexState.NORMAL:
                break

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

        """
        from pymochow.model.table import AnnSearch, HNSWSearchParams

        search_filter = None
        if query.filters is not None:
            search_filter = self._build_filter_condition(query.filters, **kwargs)
        anns = AnnSearch(
            vector_field=FIELD_VECTOR,
            vector_floats=query.query_embedding,
            params=HNSWSearchParams(ef=DEFAULT_HNSW_EF, limit=query.similarity_top_k),
            filter=search_filter,
        )
        res = self._table.search(anns=anns, retrieve_vector=True)
        rows = res.rows
        if rows is None or len(rows) == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

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
            if filter.operator:
                if filter.operator in ["<", ">", "<=", ">=", "!="]:
                    condition = f"{filter.key}{filter.operator}{filter.value}"
                elif filter.operator in ["=="]:
                    if isinstance(filter.value, str):
                        condition = f"{filter.key}='{filter.value}'"
                    else:
                        condition = f"{filter.key}=={filter.value}"
                else:
                    raise ValueError(
                        f"Filter operator {filter.operator} not supported."
                    )
            else:
                condition = f"{filter.key}={filter.value}"

            filters_list.append(condition)

        return standard_filters.condition.join(filters_list)
