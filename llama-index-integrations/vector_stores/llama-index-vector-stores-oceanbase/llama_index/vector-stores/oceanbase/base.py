"""OceanBase Vector Store."""

import logging
import json
from typing import Any, Optional, List

from llama_index.core.utils import iter_batch
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from sqlalchemy import Column, Table, String, JSON, text, func
from sqlalchemy.dialects.mysql import LONGTEXT
from pyobvector import ObVecClient, VECTOR

logger = logging.getLogger(__name__)

DEFAULT_OCEANBASE_BATCH_SIZE = 100
DEFAULT_OCEANBASE_VECTOR_TABLE_NAME = "llama_vector"
DEFAULT_OCEANBASE_HNSW_BUILD_PARAM = {"M": 16, "efConstruction": 256}
DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM = {"efSearch": 64}
OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPE = "HNSW"
DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE = "l2"

DEFAULT_OCEANBASE_PFIELD = "id"
DEFAULT_OCEANBASE_DOCID_FIELD = "doc_id"
DEFAULT_OCEANBASE_VEC_FIELD = "embedding"
DEFAULT_OCEANBASE_DOC_FIELD = "document"
DEFAULT_OCEANBASE_METADATA_FIELD = "metadata"

DEFAULT_OCEANBASE_VEC_INDEX_NAME = "vidx"


def _parse_filter_value(filter_value: any, is_text_match: bool = False):
    if filter_value is None:
        return filter_value

    if is_text_match:
        return f"'{filter_value!s}%'"

    if isinstance(filter_value, str):
        return f"'{filter_value!s}'"

    if isinstance(filter_value, list):
        return "(" + ",".join([str(v) for v in filter_value]) + ")"

    return str(filter_value)


def _to_oceanbase_filter(metadata_filters: Optional[MetadataFilters] = None) -> str:
    filters = []
    for filter in metadata_filters.filters:
        if isinstance(filter, MetadataFilters):
            filters.append(f"({_to_oceanbase_filter(filter)})")
            continue

        filter_value = _parse_filter_value(filter.value)
        if filter_value is None and filter.operator != FilterOperator.IS_EMPTY:
            continue

        if filter.operator == FilterOperator.EQ:
            filters.append(f"{filter.key}={filter_value}")
        elif filter.operator == FilterOperator.GT:
            filters.append(f"{filter.key}>{filter_value}")
        elif filter.operator == FilterOperator.LT:
            filters.append(f"{filter.key}<{filter_value}")
        elif filter.operator == FilterOperator.NE:
            filters.append(f"{filter.key}!={filter_value}")
        elif filter.operator == FilterOperator.GTE:
            filters.append(f"{filter.key}>={filter_value}")
        elif filter.operator == FilterOperator.LTE:
            filters.append(f"{filter.key}<={filter_value}")
        elif filter.operator == FilterOperator.IN:
            filters.append(f"{filter.key} in {filter_value}")
        elif filter.operator == FilterOperator.NIN:
            filters.append(f"{filter.key} not in {filter_value}")
        elif filter.operator == FilterOperator.TEXT_MATCH:
            filters.append(
                f"{filter.key} like {_parse_filter_value(filter.value, True)}"
            )
        elif filter.operator == FilterOperator.IS_EMPTY:
            filters.append(f"{filter.key} IS NULL")
        else:
            raise ValueError(
                f'Operator {filter.operator} ("{filter.operator.value}") is not supported by OceanBase.'
            )
    return f" {metadata_filters.condition.value} ".join(filters)


class OceanBaseVectorStore(BasePydanticVectorStore):
    """OceanBase Vector Store.

    You need to install `pyobvector` and run a standalone observer or OceanBase cluster.

    See the following documentation for how to deploy OceanBase:
    https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md

    IF USING L2/IP metric, IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        client (ObVecClient): OceanBase vector store client.
            Refer to `pyobvector` for more information.
        dim (int): Dimension of embedding vector.
        table_name (str): Which table name to use. Defaults to "llama_vector".
        vidx_metric_type (str): Metric method of distance between vectors.
            This parameter takes values in `l2` and `ip`. Defaults to `l2`.
        vidx_algo_params (Optional[dict]): Which index params to use. Now OceanBase
            supports HNSW only. Refer to `DEFAULT_OCEANBASE_HNSW_BUILD_PARAM`
            for example.
        drop_old (bool): Whether to drop the current table. Defaults
            to False.
        primary_field (str): Name of the primary key column. Defaults to "id".
        doc_id_field (str): Name of the doc id column. Defaults to "doc_id".
        vector_field (str): Name of the vector column. Defaults to "embedding".
        text_field (str): Name of the text column. Defaults to "document".
        metadata_field (Optional[str]): Name of the metadata column.
            Defaults to "metadata". When `metadata_field` is specified,
            the document's metadata will store as json.
        vidx_name (str): Name of the vector index table.
        partitions (ObPartition): Partition strategy of table. Refer to `pyobvector`'s
            documentation for more examples.
        extra_columns (Optional[List[Column]]): Extra sqlalchemy columns
            to add to the table.

    Examples:
        `pip install llama-index-vector-stores-oceanbase`

        ```python
        from llama_index.vector_stores.oceanbase import OceanBaseVectorStore

        # Setup ObVecClient
        from pyobvector import ObVecClient

        client = ObVecClient(
            uri=os.getenv("OB_URI", "127.0.0.1:2881"),
            user=os.getenv("OB_USER", "root@test"),
            password=os.getenv("OB_PWD", ""),
            db_name=os.getenv("OB_DBNAME", "test"),
        )

        # Initialize OceanBaseVectorStore
        oceanbase = OceanBaseVectorStore(
            client=client,
            dim=1024,
        )
        ```
    """

    stores_text: bool = True

    def __init__(
        self,
        client: ObVecClient,
        dim: int,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        vidx_metric_type: str = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        primary_field: str = DEFAULT_OCEANBASE_PFIELD,
        doc_id_field: str = DEFAULT_OCEANBASE_DOCID_FIELD,
        vector_field: str = DEFAULT_OCEANBASE_VEC_FIELD,
        text_field: str = DEFAULT_OCEANBASE_DOC_FIELD,
        metadata_field: str = DEFAULT_OCEANBASE_METADATA_FIELD,
        vidx_name: str = DEFAULT_OCEANBASE_VEC_INDEX_NAME,
        partitions: Optional[Any] = None,
        extra_columns: Optional[List[Column]] = None,
        **kwargs,
    ):
        super().__init__()

        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector=0.1.2`."
            )

        if client is not None:
            if not isinstance(client, ObVecClient):
                raise ValueError("client must be of type pyobvector.ObVecClient")
        else:
            raise ValueError("client not specified")

        self.dim = dim
        self._client: ObVecClient = client
        self.table_name = table_name
        self.extra_columns = extra_columns
        self.vidx_metric_type = vidx_metric_type.lower()
        if self.vidx_metric_type not in ("l2", "ip"):
            raise ValueError("`vidx_metric_type` should be set in `l2`/`ip`.")
        self.vidx_algo_params = vidx_algo_params or DEFAULT_OCEANBASE_HNSW_BUILD_PARAM

        self.primary_field = primary_field
        self.doc_id_field = doc_id_field
        self.vector_field = vector_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.vidx_name = vidx_name
        self.partition = partitions
        self.hnsw_ef_search = -1

        if drop_old:
            self._client.drop_table_if_exist(table_name=self.table_name)

        self._create_table_with_index()

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self.vidx_metric_type == "l2":
            return func.l2_distance
        if self.vidx_metric_type == "cosine":
            return func.cosine_distance
        if self.vidx_metric_type == "ip":
            return func.inner_product
        raise ValueError(f"Invalid vector index metric type: {self.vidx_metric_type}")

    def _load_table(self) -> None:
        table = Table(
            self.table_name,
            self._client.metadata_obj,
            autoload_with=self._client.engine,
        )
        column_names = [column.name for column in table.columns]
        optional_len = len(self.extra_columns or [])
        assert len(column_names) == (5 + optional_len)

        logging.info(f"load exist table with {column_names} columns")
        self.primary_field = column_names[0]
        self.doc_id_field = column_names[1]
        self.vector_field = column_names[2]
        self.text_field = column_names[3]
        self.metadata_field = column_names[4]

    def _create_table_with_index(self):
        if self._client.check_table_exists(self.table_name):
            self._load_table()
            return

        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.doc_id_field, String(4096)),
            Column(self.vector_field, VECTOR(self.dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
        ]
        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self._client.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPE,
            index_name=self.vidx_name,
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        self._client.create_table_with_index_params(
            table_name=self.table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            partitions=self.partition,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OceanBaseVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        batch_size: Optional[int] = None,
        extras: Optional[List[dict]] = None,
    ) -> List[str]:
        """Add nodes into OceanBase.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.
            batch_size (Optional[int]): Insert nodes in batch.
            extras (Optional[List[dict]]): If `extra_columns` is set
                when initializing `OceanBaseVectorStore`, you can add
                nodes with extra infos.

        Returns:
            List[str]: List of ids inserted.
        """
        batch_size = batch_size or DEFAULT_OCEANBASE_BATCH_SIZE

        extra_data = extras or [{} for _ in nodes]
        if len(nodes) != len(extra_data):
            raise ValueError("nodes size & extras size mismatch")

        data = [
            {
                self.primary_field: node.id_,
                self.doc_id_field: node.ref_doc_id or None,
                self.vector_field: node.get_embedding(),
                self.text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                self.metadata_field: node_to_metadata_dict(node, remove_text=True),
                **extra,
            }
            for node, extra in zip(nodes, extra_data)
        ]
        for data_batch in iter_batch(data, batch_size):
            self._client.insert(self.table_name, data_batch)
        return [node.id_ for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
        """
        self._client.delete(
            table_name=self.table_name,
            where_clause=[text(f"{self.doc_id_field}={ref_doc_id}")],
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.
        """
        if filters is not None:
            filter = _to_oceanbase_filter(filters)
        else:
            filter = None

        self._client.delete(
            table_name=self.table_name,
            ids=node_ids if node_ids is not None else None,
            where_clause=[text(filter)] if filter is not None else None,
        )

    def clear(self) -> None:
        """Clears table."""
        self._client.perform_raw_text_sql(text(f"TRUNCATE TABLE {self.table_name}"))

    def query(
        self, query: VectorStoreQuery, param: Optional[dict] = None, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Perform top-k ANN search.

        Args:
            query (VectorStoreQuery): query infos
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
        """
        search_param = (
            param if param is not None else DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM
        )
        ef_search = search_param.get(
            "efSearch", DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM["efSearch"]
        )
        if ef_search != self.hnsw_ef_search:
            self._client.set_ob_hnsw_ef_search(ef_search)
            self.hnsw_ef_search = ef_search

        res = self._client.ann_search(
            table_name=self.table_name,
            vec_data=query.query_embedding,
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            output_column_names=[
                self.primary_field,
                self.text_field,
                self.metadata_field,
            ],
            topk=query.similarity_top_k,
            where_clause=text(_to_oceanbase_filter(query.filters)),
        )

        records = []
        for r in res.fetchall():
            records.append(r)
        return VectorStoreQueryResult(
            nodes=[
                metadata_dict_to_node(
                    metadata=json.loads(r[2]),
                    text=r[1],
                )
                for r in records
            ],
            similarities=[r[3] for r in records],
            ids=[r[0] for r in records],
        )
