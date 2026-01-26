"""OceanBase Vector Store."""

import math
import logging
import json
import re
from typing import Any, Optional, List, Dict, Iterable, Tuple, Set, Literal

from llama_index.core.bridge.pydantic import (
    PrivateAttr,
    Field,
    field_validator,
    model_validator,
)
from llama_index.core.utils import iter_batch
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQueryMode,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from sqlalchemy import Column, Table, String, JSON, text, func, select, bindparam
from sqlalchemy.dialects.mysql import LONGTEXT
from pyobvector import (
    ObVecClient,
    VECTOR,
    SPARSE_VECTOR,
    FtsIndexParam,
    FtsParser,
    MatchAgainst,
)
from pyobvector.client.index_param import VecIndexType
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OCEANBASE_BATCH_SIZE = 100
DEFAULT_OCEANBASE_VECTOR_TABLE_NAME = "llama_vector"
DEFAULT_OCEANBASE_HNSW_BUILD_PARAM = {"M": 16, "efConstruction": 256}
DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM = {"efSearch": 64}
DEFAULT_OCEANBASE_IVF_BUILD_PARAM = {"nlist": 128}
DEFAULT_OCEANBASE_FLAT_BUILD_PARAM = {"nlist": 1}
OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES = {
    "HNSW": VecIndexType.HNSW,
    "HNSW_SQ": VecIndexType.HNSW_SQ,
    "IVF": VecIndexType.IVFFLAT,
    "IVF_FLAT": VecIndexType.IVFFLAT,
    "IVF_SQ": VecIndexType.IVFSQ,
    "IVF_PQ": VecIndexType.IVFPQ,
    "FLAT": VecIndexType.IVFFLAT,
}
DEFAULT_OCEANBASE_INDEX_TYPE = "HNSW"
DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE = "l2"

DEFAULT_OCEANBASE_PFIELD = "id"
DEFAULT_OCEANBASE_DOCID_FIELD = "doc_id"
DEFAULT_OCEANBASE_VEC_FIELD = "embedding"
DEFAULT_OCEANBASE_DOC_FIELD = "document"
DEFAULT_OCEANBASE_METADATA_FIELD = "metadata"
DEFAULT_OCEANBASE_SPARSE_VECTOR_FIELD = "sparse_embedding"
DEFAULT_OCEANBASE_FULLTEXT_FIELD = "fulltext_content"

DEFAULT_OCEANBASE_VEC_INDEX_NAME = "vidx"

VidxMetricType = Literal["l2", "inner_product", "cosine"]
IndexType = Literal[
    "HNSW",
    "HNSW_SQ",
    "IVF",
    "IVF_FLAT",
    "IVF_SQ",
    "IVF_PQ",
    "FLAT",
]


_JSON_PATH_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_]+$")


def _escape_json_path_segment(segment: str) -> str:
    if not segment:
        raise ValueError("Metadata filter key segment cannot be empty")
    if _JSON_PATH_SAFE_SEGMENT.match(segment):
        return segment
    escaped = segment.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_text_clause(sql: str, params: Dict[str, Any], expanding_params: Set[str]):
    clause = text(sql)
    for name, value in params.items():
        if name in expanding_params:
            clause = clause.bindparams(bindparam(name, value, expanding=True))
        else:
            clause = clause.bindparams(bindparam(name, value))
    return clause


def _euclidean_similarity(distance: float) -> float:
    return 1.0 - distance / math.sqrt(2)


def _neg_inner_product_similarity(distance: float) -> float:
    return -distance


def _cosine_similarity(distance: float) -> float:
    return 1.0 - distance


def _normalize(vector: List[float]) -> List[float]:
    arr = np.array(vector)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vector
    arr = arr / norm
    return arr.tolist()


class OceanBaseVectorStore(BasePydanticVectorStore):
    """
    OceanBase Vector Store.

    You need to install `pyobvector` and run a standalone observer or OceanBase cluster.

    See the following documentation for how to deploy OceanBase:
    https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md

    IF USING L2/INNER_PRODUCT/COSINE metric, IT IS HIGHLY SUGGESTED TO set `normalize = True`.

    Args:
        _client (ObVecClient): OceanBase vector store client.
            Refer to `pyobvector` for more information.
        dim (int): Dimension of embedding vector.
        table_name (str): Which table name to use. Defaults to "llama_vector".
        vidx_metric_type (str): Metric method of distance between vectors.
            This parameter takes values in `l2`, `inner_product`, and `cosine`.
            Defaults to `l2`.
        vidx_algo_params (Optional[dict]): Which index params to use. OceanBase
            supports multiple index types. Refer to the default build params
            for example.
        index_type (str): Vector index type. Supports "HNSW", "HNSW_SQ", "IVF",
            "IVF_FLAT", "IVF_SQ", "IVF_PQ", "FLAT". Defaults to "HNSW".
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
        include_sparse (bool): Enable sparse vector support. Defaults to False.
        include_fulltext (bool): Enable full-text search support. Defaults to False.
        sparse_vector_field (str): Name of the sparse vector column.
        fulltext_field (str): Name of the fulltext column.
        normalize (bool): normalize vector or not.

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

    vidx_metric_type: VidxMetricType = Field(
        default=DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        description="Metric method of distance between vectors.",
    )
    index_type: IndexType = Field(
        default=DEFAULT_OCEANBASE_INDEX_TYPE,
        description="Vector index type.",
    )
    include_sparse: bool = Field(
        default=False,
        description="Enable sparse vector support.",
    )
    include_fulltext: bool = Field(
        default=False,
        description="Enable full-text search support.",
    )
    sparse_vector_field: str = Field(
        default=DEFAULT_OCEANBASE_SPARSE_VECTOR_FIELD,
        description="Name of the sparse vector column.",
    )
    fulltext_field: str = Field(
        default=DEFAULT_OCEANBASE_FULLTEXT_FIELD,
        description="Name of the fulltext column.",
    )
    normalize: bool = Field(
        default=False,
        description="Normalize vector or not.",
    )

    _client: ObVecClient = PrivateAttr()
    _dim: int = PrivateAttr()
    _table_name: str = PrivateAttr()
    _vidx_metric_type: str = PrivateAttr()
    _vidx_algo_params: dict = PrivateAttr()
    _index_type: str = PrivateAttr()
    _primary_field: str = PrivateAttr()
    _doc_id_field: str = PrivateAttr()
    _vector_field: str = PrivateAttr()
    _text_field: str = PrivateAttr()
    _metadata_field: str = PrivateAttr()
    _vidx_name: str = PrivateAttr()
    _partitions: Optional[Any] = PrivateAttr()
    _extra_columns: Optional[List[Column]] = PrivateAttr()
    _include_sparse: bool = PrivateAttr()
    _include_fulltext: bool = PrivateAttr()
    _sparse_vector_field: str = PrivateAttr()
    _fulltext_field: str = PrivateAttr()
    _hnsw_ef_search: int = PrivateAttr()
    _normalize: bool = PrivateAttr()

    @field_validator("vidx_metric_type", mode="before")
    @classmethod
    def _normalize_vidx_metric_type(cls, value: str) -> str:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("index_type", mode="before")
    @classmethod
    def _normalize_index_type(cls, value: str) -> str:
        if isinstance(value, str):
            return value.upper()
        return value

    @model_validator(mode="after")
    def _validate_fulltext_support(self) -> "OceanBaseVectorStore":
        if self.include_fulltext and not self.include_sparse:
            raise ValueError(
                "Full-text search requires sparse vector support. "
                "Set include_sparse=True when include_fulltext=True."
            )
        return self

    def __init__(
        self,
        client: ObVecClient,
        dim: int,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        vidx_metric_type: VidxMetricType = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        index_type: IndexType = DEFAULT_OCEANBASE_INDEX_TYPE,
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
        include_sparse: bool = False,
        include_fulltext: bool = False,
        sparse_vector_field: str = DEFAULT_OCEANBASE_SPARSE_VECTOR_FIELD,
        fulltext_field: str = DEFAULT_OCEANBASE_FULLTEXT_FIELD,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(
            vidx_metric_type=vidx_metric_type,
            index_type=index_type,
            include_sparse=include_sparse,
            include_fulltext=include_fulltext,
            sparse_vector_field=sparse_vector_field,
            fulltext_field=fulltext_field,
            normalize=normalize,
        )

        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        if client is not None:
            if not isinstance(client, ObVecClient):
                raise ValueError("client must be of type pyobvector.ObVecClient")
        else:
            raise ValueError("client not specified")

        self._dim = dim
        self._client: ObVecClient = client
        self._table_name = table_name
        self._extra_columns = extra_columns
        self._include_sparse = self.include_sparse
        self._include_fulltext = self.include_fulltext
        self._sparse_vector_field = self.sparse_vector_field
        self._fulltext_field = self.fulltext_field
        self._vidx_metric_type = self.vidx_metric_type
        self._index_type = self.index_type
        if vidx_algo_params is None:
            if self._index_type in ("HNSW", "HNSW_SQ"):
                self._vidx_algo_params = DEFAULT_OCEANBASE_HNSW_BUILD_PARAM
            elif self._index_type in ("IVF", "IVF_FLAT", "IVF_SQ", "IVF_PQ"):
                self._vidx_algo_params = DEFAULT_OCEANBASE_IVF_BUILD_PARAM
            elif self._index_type == "FLAT":
                self._vidx_algo_params = DEFAULT_OCEANBASE_FLAT_BUILD_PARAM
            else:
                self._vidx_algo_params = {}
        else:
            self._vidx_algo_params = vidx_algo_params

        self._primary_field = primary_field
        self._doc_id_field = doc_id_field
        self._vector_field = vector_field
        self._text_field = text_field
        self._metadata_field = metadata_field
        self._vidx_name = vidx_name
        self._partitions = partitions
        self._hnsw_ef_search = -1
        self._normalize = normalize

        if drop_old:
            self._client.drop_table_if_exist(table_name=self._table_name)

        self._create_table_with_index()

    def _enhance_filter_key(self, filter_key: str) -> str:
        segments = filter_key.split(".")
        json_path = "$." + ".".join(_escape_json_path_segment(seg) for seg in segments)
        return f"{self._metadata_field}->'{json_path}'"

    def _add_filter_param(
        self,
        params: Dict[str, Any],
        expanding_params: Set[str],
        value: Any,
        *,
        prefix: str,
        expanding: bool = False,
    ) -> str:
        name = f"{prefix}_{len(params)}"
        params[name] = value
        if expanding:
            expanding_params.add(name)
        return f":{name}"

    def _to_oceanbase_filter(
        self,
        metadata_filters: Optional[MetadataFilters] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        expanding_params: Optional[Set[str]] = None,
    ) -> str:
        if metadata_filters is None:
            return ""
        if params is None:
            params = {}
        if expanding_params is None:
            expanding_params = set()

        filters = []
        for filter in metadata_filters.filters:
            if isinstance(filter, MetadataFilters):
                nested = self._to_oceanbase_filter(
                    filter, params=params, expanding_params=expanding_params
                )
                if nested:
                    filters.append(f"({nested})")
                continue

            if filter.operator == FilterOperator.IS_EMPTY:
                filters.append(f"{self._enhance_filter_key(filter.key)} IS NULL")
                continue

            if filter.value is None:
                continue

            column = self._enhance_filter_key(filter.key)
            if filter.operator == FilterOperator.EQ:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="eq"
                )
                filters.append(f"{column}={placeholder}")
            elif filter.operator == FilterOperator.GT:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="gt"
                )
                filters.append(f"{column}>{placeholder}")
            elif filter.operator == FilterOperator.LT:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="lt"
                )
                filters.append(f"{column}<{placeholder}")
            elif filter.operator == FilterOperator.NE:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="ne"
                )
                filters.append(f"{column}!={placeholder}")
            elif filter.operator == FilterOperator.GTE:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="gte"
                )
                filters.append(f"{column}>={placeholder}")
            elif filter.operator == FilterOperator.LTE:
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="lte"
                )
                filters.append(f"{column}<={placeholder}")
            elif filter.operator in (FilterOperator.IN, FilterOperator.NIN):
                if not isinstance(filter.value, list):
                    raise ValueError(
                        f"Operator {filter.operator.value} expects a list value."
                    )
                if len(filter.value) == 0:
                    filters.append(
                        "1=0" if filter.operator == FilterOperator.IN else "1=1"
                    )
                    continue
                placeholder = self._add_filter_param(
                    params, expanding_params, filter.value, prefix="in", expanding=True
                )
                op = "in" if filter.operator == FilterOperator.IN else "not in"
                filters.append(f"{column} {op} {placeholder}")
            elif filter.operator == FilterOperator.TEXT_MATCH:
                placeholder = self._add_filter_param(
                    params,
                    expanding_params,
                    f"{filter.value}%",
                    prefix="like",
                )
                filters.append(f"{column} like {placeholder}")
            else:
                raise ValueError(
                    f'Operator {filter.operator} ("{filter.operator.value}") is not supported by OceanBase.'
                )

        if not filters:
            return ""
        condition = metadata_filters.condition or FilterCondition.AND
        if condition == FilterCondition.NOT:
            return f"NOT ({' AND '.join(filters)})"
        return f" {condition.value} ".join(filters)

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self._vidx_metric_type == "l2":
            return func.l2_distance
        if self._vidx_metric_type == "cosine":
            return func.cosine_distance
        if self._vidx_metric_type == "inner_product":
            return func.negative_inner_product
        raise ValueError(f"Invalid vector index metric type: {self._vidx_metric_type}")

    def _load_table(self) -> None:
        table = Table(
            self._table_name,
            self._client.metadata_obj,
            autoload_with=self._client.engine,
        )
        column_names = [column.name for column in table.columns]
        required_columns = [
            self._primary_field,
            self._doc_id_field,
            self._vector_field,
            self._text_field,
            self._metadata_field,
        ]
        missing = [name for name in required_columns if name not in column_names]
        if missing:
            raise ValueError(
                "Existing table is missing required columns: "
                f"{missing}. Found columns: {column_names}"
            )

        logging.info(f"load exist table with {column_names} columns")
        if self._include_sparse:
            if self._sparse_vector_field not in column_names:
                raise ValueError(
                    f"Sparse vector column is missing in table {self._table_name}"
                )
        if self._include_fulltext:
            if self._fulltext_field not in column_names:
                raise ValueError(
                    f"Fulltext column is missing in table {self._table_name}"
                )

    def _create_table_with_index(self):
        if self._client.check_table_exists(self._table_name):
            self._load_table()
            return

        cols = [
            Column(
                self._primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self._doc_id_field, String(4096)),
            Column(self._vector_field, VECTOR(self._dim)),
            Column(self._text_field, LONGTEXT),
            Column(self._metadata_field, JSON),
        ]
        if self._include_sparse:
            cols.append(Column(self._sparse_vector_field, SPARSE_VECTOR()))
        if self._include_fulltext:
            cols.append(Column(self._fulltext_field, LONGTEXT))
        if self._extra_columns is not None:
            cols.extend(self._extra_columns)

        vidx_params = self._client.prepare_index_params()
        vidx_params.add_index(
            field_name=self._vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES[self._index_type],
            index_name=self._vidx_name,
            metric_type=self._vidx_metric_type,
            params=self._vidx_algo_params,
        )
        if self._include_sparse:
            sparse_index_kwargs = {}
            if self._client._is_seekdb():
                sparse_index_kwargs["sparse_index_type"] = "sindi"
            vidx_params.add_index(
                field_name=self._sparse_vector_field,
                index_type=VecIndexType.DAAT,
                index_name=f"{self._vidx_name}_sparse",
                metric_type="inner_product",
                **sparse_index_kwargs,
            )

        fts_idxs = None
        if self._include_fulltext:
            fts_idxs = [
                FtsIndexParam(
                    index_name=f"{self._vidx_name}_fts",
                    field_names=[self._fulltext_field],
                    parser_type=FtsParser.NGRAM,
                )
            ]

        self._client.create_table_with_index_params(
            table_name=self._table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            fts_idxs=fts_idxs,
            partitions=self._partitions,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OceanBaseVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Get nodes from OceanBase.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.

        Returns:
            List[BaseNode]: List of text nodes.

        """
        where_clause = self._build_where_clause(filters=filters)

        res = self._client.get(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[where_clause] if where_clause is not None else None,
            output_column_name=[
                self._text_field,
                self._metadata_field,
            ],
        )

        return [
            metadata_dict_to_node(
                metadata=(json.loads(r[1]) if not isinstance(r[1], dict) else r[1]),
                text=r[0],
            )
            for r in res.fetchall()
        ]

    def add(
        self,
        nodes: List[BaseNode],
        batch_size: Optional[int] = None,
        extras: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Add nodes into OceanBase.

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
                self._primary_field: node.id_,
                self._doc_id_field: node.ref_doc_id or None,
                self._vector_field: (
                    node.get_embedding()
                    if not self._normalize
                    else _normalize(node.get_embedding())
                ),
                self._text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                self._metadata_field: node_to_metadata_dict(node, remove_text=True),
                **extra,
            }
            for node, extra in zip(nodes, extra_data)
        ]
        for data_batch in iter_batch(data, batch_size):
            self._client.insert(self._table_name, data_batch)
        return [node.id_ for node in nodes]

    def _add_nodes_with_hybrid_fields(
        self,
        nodes: List[BaseNode],
        *,
        sparse_embeddings: Optional[List[Dict[int, float]]] = None,
        fulltext_content: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if sparse_embeddings is None and fulltext_content is None:
            raise ValueError("At least one hybrid field must be provided")

        if sparse_embeddings is not None and not self._include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when initializing."
            )
        if fulltext_content is not None and not self._include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True when initializing."
            )

        if sparse_embeddings is not None and len(nodes) != len(sparse_embeddings):
            raise ValueError("Number of nodes must match number of sparse embeddings")
        if fulltext_content is not None and len(nodes) != len(fulltext_content):
            raise ValueError("Number of nodes must match number of fulltext items")

        if node_ids is None:
            node_ids = [node.id_ for node in nodes]
        if len(node_ids) != len(nodes):
            raise ValueError("Number of node_ids must match number of nodes")

        batch_size = batch_size or DEFAULT_OCEANBASE_BATCH_SIZE

        data: List[dict] = []
        for idx, node in enumerate(nodes):
            record = {
                self._primary_field: node_ids[idx],
                self._doc_id_field: node.ref_doc_id or None,
                self._vector_field: (
                    node.get_embedding()
                    if not self._normalize
                    else _normalize(node.get_embedding())
                ),
                self._text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                self._metadata_field: node_to_metadata_dict(node, remove_text=True),
            }
            if sparse_embeddings is not None:
                record[self._sparse_vector_field] = sparse_embeddings[idx]
            if fulltext_content is not None:
                record[self._fulltext_field] = fulltext_content[idx]
            data.append(record)

        for data_batch in iter_batch(data, batch_size):
            self._client.upsert(self._table_name, data_batch)
        return node_ids

    def add_sparse_nodes(
        self,
        nodes: List[BaseNode],
        sparse_embeddings: List[Dict[int, float]],
        *,
        batch_size: Optional[int] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[str]:
        return self._add_nodes_with_hybrid_fields(
            nodes,
            sparse_embeddings=sparse_embeddings,
            batch_size=batch_size,
            node_ids=node_ids,
        )

    def add_nodes_with_fulltext(
        self,
        nodes: List[BaseNode],
        fulltext_content: List[str],
        *,
        batch_size: Optional[int] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[str]:
        return self._add_nodes_with_hybrid_fields(
            nodes,
            fulltext_content=fulltext_content,
            batch_size=batch_size,
            node_ids=node_ids,
        )

    def add_nodes_with_hybrid_fields(
        self,
        nodes: List[BaseNode],
        *,
        sparse_embeddings: Optional[List[Dict[int, float]]] = None,
        fulltext_content: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if sparse_embeddings is None and fulltext_content is None:
            return self.add(nodes, batch_size=batch_size)
        return self._add_nodes_with_hybrid_fields(
            nodes,
            sparse_embeddings=sparse_embeddings,
            fulltext_content=fulltext_content,
            batch_size=batch_size,
            node_ids=node_ids,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._client.delete(
            table_name=self._table_name,
            where_clause=[
                _build_text_clause(
                    f"{self._doc_id_field}=:ref_doc_id",
                    {"ref_doc_id": ref_doc_id},
                    set(),
                )
            ],
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Deletes nodes.

        Args:
            node_ids (Optional[List[str]], optional): IDs of nodes to delete.
                Defaults to None.
            filters (Optional[MetadataFilters], optional): Metadata filters.
                Defaults to None.

        """
        where_clause = self._build_where_clause(filters=filters, node_ids=node_ids)

        self._client.delete(
            table_name=self._table_name,
            ids=node_ids,
            where_clause=[where_clause] if where_clause is not None else None,
        )

    def clear(self) -> None:
        """Clears table."""
        self._client.perform_raw_text_sql(f"TRUNCATE TABLE {self._table_name}")

    def _parse_distance_to_similarities(self, distance: float) -> float:
        if self._vidx_metric_type == "l2":
            return _euclidean_similarity(distance)
        elif self._vidx_metric_type == "inner_product":
            return _neg_inner_product_similarity(distance)
        elif self._vidx_metric_type == "cosine":
            return _cosine_similarity(distance)
        raise ValueError(f"Metric Type {self._vidx_metric_type} is not supported")

    def _build_where_clause(
        self,
        filters: Optional[MetadataFilters] = None,
        doc_ids: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
    ):
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        expanding_params: Set[str] = set()

        filter_clause = self._to_oceanbase_filter(
            filters, params=params, expanding_params=expanding_params
        )
        if filter_clause:
            clauses.append(filter_clause)

        if doc_ids is not None:
            if len(doc_ids) == 0:
                return text("1=0")
            placeholder = self._add_filter_param(
                params, expanding_params, doc_ids, prefix="doc_ids", expanding=True
            )
            clauses.append(f"{self._doc_id_field} in {placeholder}")

        if node_ids is not None:
            if len(node_ids) == 0:
                return text("1=0")
            placeholder = self._add_filter_param(
                params, expanding_params, node_ids, prefix="node_ids", expanding=True
            )
            clauses.append(f"{self._primary_field} in {placeholder}")

        if not clauses:
            return None
        return _build_text_clause(" AND ".join(clauses), params, expanding_params)

    def _handle_hnsw_ef_search(self, search_param: dict) -> None:
        if self._index_type in ("HNSW", "HNSW_SQ"):
            ef_search = search_param.get(
                "efSearch", DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM["efSearch"]
            )
            if ef_search != self._hnsw_ef_search:
                self._client.set_ob_hnsw_ef_search(ef_search)
                self._hnsw_ef_search = ef_search

    def _rows_to_records(
        self,
        rows: Iterable[Tuple[Any, ...]],
        score_fn,
        modality: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for row in rows:
            metadata_raw = row[2]
            if metadata_raw is None:
                metadata = {}
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = json.loads(metadata_raw)
            node = metadata_dict_to_node(metadata=metadata, text=row[1])
            records.append(
                {
                    "id": row[0],
                    "node": node,
                    "score": score_fn(row[3]),
                    "modality": modality,
                }
            )
        return records

    def _query_dense_records(
        self,
        query: VectorStoreQuery,
        search_param: dict,
        where_clause: Optional[Any],
    ) -> List[Dict[str, Any]]:
        self._handle_hnsw_ef_search(search_param)
        res = self._client.ann_search(
            table_name=self._table_name,
            vec_data=(
                query.query_embedding
                if not self._normalize
                else _normalize(query.query_embedding)
            ),
            vec_column_name=self._vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field,
            ],
            topk=query.similarity_top_k,
            where_clause=([where_clause] if where_clause is not None else None),
        )
        return self._rows_to_records(
            res.fetchall(), self._parse_distance_to_similarities, "vector"
        )

    def _query_sparse_records(
        self,
        sparse_query: Dict[int, float],
        top_k: int,
        where_clause: Optional[Any],
    ) -> List[Dict[str, Any]]:
        if not self._include_sparse:
            raise ValueError(
                "Sparse vector support not enabled. Set include_sparse=True when initializing."
            )
        res = self._client.ann_search(
            table_name=self._table_name,
            vec_data=sparse_query,
            vec_column_name=self._sparse_vector_field,
            distance_func=func.negative_inner_product,
            with_dist=True,
            output_column_names=[
                self._primary_field,
                self._text_field,
                self._metadata_field,
            ],
            topk=top_k,
            where_clause=([where_clause] if where_clause is not None else None),
        )
        return self._rows_to_records(
            res.fetchall(), _neg_inner_product_similarity, "sparse"
        )

    def _query_fulltext_records(
        self,
        fulltext_query: str,
        top_k: int,
        where_clause: Optional[Any],
    ) -> List[Dict[str, Any]]:
        if not self._include_fulltext:
            raise ValueError(
                "Full-text search support not enabled. Set include_fulltext=True when initializing."
            )
        table = Table(
            self._table_name,
            self._client.metadata_obj,
            autoload_with=self._client.engine,
        )
        match_expr = MatchAgainst(fulltext_query, table.c[self._fulltext_field])
        stmt = select(
            table.c[self._primary_field],
            table.c[self._text_field],
            table.c[self._metadata_field],
            match_expr.label("score"),
        ).where(match_expr)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        stmt = stmt.order_by(match_expr.desc()).limit(top_k)
        with self._client.engine.connect() as conn:
            with conn.begin():
                res = conn.execute(stmt)
                return self._rows_to_records(
                    res.fetchall(), lambda score: float(score or 0.0), "fulltext"
                )

    def _normalize_hybrid_weights(
        self,
        modalities: List[str],
        alpha: Optional[float],
    ) -> Dict[str, float]:
        base_weights = {"vector": 0.5, "sparse": 0.3, "fulltext": 0.2}
        if alpha is not None and set(modalities) == {"vector", "fulltext"}:
            base_weights = {"vector": alpha, "fulltext": 1.0 - alpha}

        weights = {k: base_weights[k] for k in modalities if k in base_weights}
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _fuse_hybrid_records(
        self,
        records_by_modality: Dict[str, List[Dict[str, Any]]],
        top_k: int,
        alpha: Optional[float],
    ) -> VectorStoreQueryResult:
        modalities = [m for m, records in records_by_modality.items() if records]
        if not modalities:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        weights = self._normalize_hybrid_weights(modalities, alpha)
        priority = {"vector": 0, "sparse": 1, "fulltext": 2}

        combined: Dict[str, Dict[str, Any]] = {}
        for modality, records in records_by_modality.items():
            if not records:
                continue
            weight = weights.get(modality, 0.0)
            count = len(records)
            for idx, record in enumerate(records):
                rank_score = 1.0 - (idx / count) if count > 0 else 0.0
                combined_score = weight * rank_score
                existing = combined.get(record["id"])
                if existing is None:
                    combined[record["id"]] = {
                        "node": record["node"],
                        "score": combined_score,
                        "modality": modality,
                    }
                else:
                    existing["score"] += combined_score
                    if priority[modality] < priority[existing["modality"]]:
                        existing["node"] = record["node"]
                        existing["modality"] = modality

        ranked = sorted(
            combined.items(), key=lambda item: item[1]["score"], reverse=True
        )
        ranked = ranked[:top_k]
        nodes = [entry["node"] for _, entry in ranked]
        similarities = [entry["score"] for _, entry in ranked]
        ids = [doc_id for doc_id, _ in ranked]
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _records_to_query_result(
        self, records: List[Dict[str, Any]]
    ) -> VectorStoreQueryResult:
        return VectorStoreQueryResult(
            nodes=[record["node"] for record in records],
            similarities=[record["score"] for record in records],
            ids=[record["id"] for record in records],
        )

    def query(
        self, query: VectorStoreQuery, param: Optional[dict] = None, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Perform top-k ANN search.

        Args:
            query (VectorStoreQuery): query infos
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
            sparse_query (Optional[Dict[int, float]]): Sparse vector query data for
                `VectorStoreQueryMode.SPARSE` or `VectorStoreQueryMode.HYBRID`.
            fulltext_query (Optional[str]): Fulltext query string for
                `VectorStoreQueryMode.TEXT_SEARCH` or `VectorStoreQueryMode.HYBRID`.

        """
        search_param = (
            param if param is not None else DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM
        )
        qfilters = self._build_where_clause(
            query.filters, query.doc_ids, query.node_ids
        )

        if query.mode == VectorStoreQueryMode.DEFAULT:
            if query.query_embedding is None:
                raise ValueError(
                    "query_embedding must be provided for OceanBase search"
                )
            records = self._query_dense_records(query, search_param, qfilters)
            return self._records_to_query_result(records)

        if query.mode == VectorStoreQueryMode.SPARSE:
            sparse_query = kwargs.get("sparse_query")
            if sparse_query is None:
                raise ValueError("sparse_query must be provided for sparse search")
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            records = self._query_sparse_records(sparse_query, sparse_top_k, qfilters)
            return self._records_to_query_result(records)

        if query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            fulltext_query = kwargs.get("fulltext_query", query.query_str)
            if fulltext_query is None:
                raise ValueError(
                    "query_str or fulltext_query must be provided for text search"
                )
            records = self._query_fulltext_records(
                fulltext_query, query.similarity_top_k, qfilters
            )
            return self._records_to_query_result(records)

        if query.mode == VectorStoreQueryMode.HYBRID:
            fulltext_query = kwargs.get("fulltext_query", query.query_str)
            sparse_query = kwargs.get("sparse_query")
            records_by_modality: Dict[str, List[Dict[str, Any]]] = {}
            if query.query_embedding is not None:
                records_by_modality["vector"] = self._query_dense_records(
                    query, search_param, qfilters
                )
            if sparse_query is not None:
                sparse_top_k = query.sparse_top_k or query.similarity_top_k
                records_by_modality["sparse"] = self._query_sparse_records(
                    sparse_query, sparse_top_k, qfilters
                )
            if fulltext_query is not None:
                records_by_modality["fulltext"] = self._query_fulltext_records(
                    fulltext_query, query.similarity_top_k, qfilters
                )
            if not records_by_modality:
                raise ValueError(
                    "Hybrid search requires at least one of query_embedding, "
                    "query_str/fulltext_query, or sparse_query."
                )
            top_k = query.hybrid_top_k or query.similarity_top_k
            return self._fuse_hybrid_records(records_by_modality, top_k, query.alpha)

        raise ValueError(f"Invalid query mode: {query.mode}")
