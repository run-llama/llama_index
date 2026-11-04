import asyncio
import enum
import json
import logging
import re
import struct
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import ydb
import ydb_dbapi
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = "llamaindex_nodes"
DEFAULT_EMBED_DIM = 1536
DEFAULT_BATCH_SIZE = 100
DEFAULT_INDEX_NAME = "llamaindex_vector_index"
DEFAULT_INDEX_LEVELS = 2
DEFAULT_INDEX_CLUSTERS = 128
DEFAULT_INDEX_TREE_SEARCH_TOP_SIZE = 1
DEFAULT_INDEX_READY_TIMEOUT = 3600.0

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FLOAT_VECTOR_TAG = b"\x01"  # last byte of a YDB binary vector: 1 = Float


class YDBSearchStrategy(str, enum.Enum):
    """Search metric (the Knn::<value> function in YQL)."""

    COSINE_SIMILARITY = "CosineSimilarity"
    INNER_PRODUCT_SIMILARITY = "InnerProductSimilarity"
    COSINE_DISTANCE = "CosineDistance"
    EUCLIDEAN_DISTANCE = "EuclideanDistance"
    MANHATTAN_DISTANCE = "ManhattanDistance"

    def __str__(self) -> str:
        return self.value


DEFAULT_STRATEGY = YDBSearchStrategy.COSINE_SIMILARITY


def _vector_to_bytes(vector: Sequence[float]) -> bytes:
    """Pack a float32 vector into the YDB binary format: values followed by a type byte."""
    return struct.pack(f"<{len(vector)}f", *vector) + _FLOAT_VECTOR_TAG


def _bytes_to_vector(data: bytes) -> List[float]:
    """Inverse of _vector_to_bytes."""
    if not data or data[-1:] != _FLOAT_VECTOR_TAG:
        raise ValueError("Not a YDB float vector: missing trailing type byte 0x01")
    payload = data[:-1]
    if len(payload) % 4:
        raise ValueError("Corrupted YDB float vector: length is not a multiple of 4")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def _parse_metadata(value: Any) -> Dict[str, Any]:
    """A JSON column arrives either as a dict (native JSON) or as a string; handle both."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return json.loads(value)


def _strategy_sort_order(strategy: YDBSearchStrategy) -> str:
    """Similarities: the larger the better (DESC). Distances: the smaller the better (ASC)."""
    return "DESC" if strategy.value.endswith("Similarity") else "ASC"


def _strategy_index_param(strategy: YDBSearchStrategy) -> str:
    """Metric parameter of WITH (...) when creating a vector index."""
    return {
        YDBSearchStrategy.COSINE_SIMILARITY: "similarity=cosine",
        YDBSearchStrategy.INNER_PRODUCT_SIMILARITY: "similarity=inner_product",
        YDBSearchStrategy.COSINE_DISTANCE: "distance=cosine",
        YDBSearchStrategy.EUCLIDEAN_DISTANCE: "distance=euclidean",
        YDBSearchStrategy.MANHATTAN_DISTANCE: "distance=manhattan",
    }[strategy]


def _score_to_similarity(strategy: YDBSearchStrategy, score: Optional[float]) -> float:
    """
    LlamaIndex expects a similarity: the larger, the closer.

    Similarities are returned as is. CosineDistance becomes 1 - d, other distances become -d.
    """
    if score is None:
        return 0.0
    score = float(score)
    if strategy.value.endswith("Similarity"):
        return score
    if strategy == YDBSearchStrategy.COSINE_DISTANCE:
        return 1.0 - score
    return -score


def _node_to_row(node: BaseNode, embed_dim: int) -> Dict[str, Any]:
    """Convert a LlamaIndex node to a table row (keys match the UPSERT struct members)."""
    embedding = node.get_embedding()
    if len(embedding) != embed_dim:
        raise ValueError(
            f"Node {node.node_id}: embedding has {len(embedding)} dimensions, "
            f"but the store is configured with embed_dim={embed_dim}"
        )
    return {
        "node_id": node.node_id,
        "ref_doc_id": node.ref_doc_id,  # None if the node has no SOURCE relationship
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "metadata": json.dumps(
            node_to_metadata_dict(node, remove_text=True, flat_metadata=False),
            ensure_ascii=False,
        ),
        "embedding": _vector_to_bytes(embedding),
    }


def _row_to_node(row: Dict[str, Any]) -> BaseNode:
    """Convert a table row (node_id, text, metadata) back to a node."""
    metadata = _parse_metadata(row["metadata"])
    node = metadata_dict_to_node(metadata)
    node.set_content(str(row["text"]))

    return node


def _rows_to_result(
    rows: List[Dict[str, Any]],
    strategy: YDBSearchStrategy = DEFAULT_STRATEGY,
) -> VectorStoreQueryResult:
    """Convert search rows to a query result (score is converted to a similarity)."""
    nodes: List[BaseNode] = []
    similarities: List[float] = []
    ids: List[str] = []
    for row in rows:
        nodes.append(_row_to_node(row))
        similarities.append(_score_to_similarity(strategy, row.get("score")))
        ids.append(row["node_id"])
    return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


def _rows_to_nodes(
    rows: List[Dict[str, Any]], node_ids: Sequence[str]
) -> List[BaseNode]:
    """Convert get_nodes rows to nodes with embeddings, in requested id order, without duplicates."""
    by_id: Dict[str, BaseNode] = {}
    for row in rows:
        node = _row_to_node(row)
        raw = row.get("embedding")
        if raw:
            node.embedding = _bytes_to_vector(raw)
        by_id[row["node_id"]] = node
    ordered: List[BaseNode] = []
    seen = set()
    for node_id in node_ids:
        if node_id in by_id and node_id not in seen:
            seen.add(node_id)
            ordered.append(by_id[node_id])
    return ordered


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Split a sequence into consecutive chunks of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _find_index(indexes: Sequence[Any], name: str) -> Optional[Any]:
    """Find an index by name in describe_table(...).indexes."""
    for index in indexes:
        if index.name == name:
            return index
    return None


def _validate_identifier(name: str, what: str) -> str:
    """Allow only letters, digits and underscores, since names are put into YQL text."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid {what}: {name!r}. Use letters, digits and underscores only."
        )
    return name


def _positive_int(value: Any, what: str) -> int:
    """Return the value if it is a positive integer, otherwise raise ValueError."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def _rename_item(temp_name: str, final_name: str) -> Any:
    """Rename an index, replacing the destination if it exists."""
    return ydb.RenameIndexItem(
        source_name=temp_name,
        destination_name=final_name,
        replace_destination=True,
    )


class YDBVectorStore(BasePydanticVectorStore):
    """
    Vector store backed by YDB.

    Search is exact (full scan) by default. For large tables you can build a vector
    index (``rebuild_vector_index``) and set ``index_enabled=True``: queries then go
    through the index.

    Async operations on a single YDBVectorStore instance must be executed within the same event loop.
    Do not reuse the instance across separate asyncio.run() calls. Create a new instance for each event loop.

    For now approximate search, metadata filters are not available.

    Examples:
        ```python
        from llama_index.vector_stores.ydb import YDBVectorStore

        vector_store = YDBVectorStore(
            host="localhost", port=2136, database="/local",
            table_name="my_nodes", embed_dim=384,
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    host: str
    port: int
    database: str
    secure: bool
    table_name: str
    embed_dim: int
    perform_setup: bool
    batch_size: int
    strategy: YDBSearchStrategy
    index_enabled: bool
    index_name: str
    index_config_levels: int
    index_config_clusters: int
    index_tree_search_top_size: int
    credentials: Any = Field(default=None, exclude=True)

    _connection: Any = PrivateAttr(default=None)
    _aloop: Any = PrivateAttr(default=None)
    _aconnection: Any = PrivateAttr(default=None)
    _aindex_lock: Any = PrivateAttr(default=None)
    _aconnection_lock: Any = PrivateAttr(default=None)
    _table_ready: bool = PrivateAttr(default=False)
    _index_setup_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _index_ready: bool = PrivateAttr(default=False)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2136,
        database: str = "/local",
        secure: bool = False,
        credentials: Any = None,
        table_name: str = DEFAULT_TABLE_NAME,
        embed_dim: int = DEFAULT_EMBED_DIM,
        perform_setup: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        strategy: Union[str, YDBSearchStrategy] = DEFAULT_STRATEGY,
        index_enabled: bool = False,
        index_name: str = DEFAULT_INDEX_NAME,
        index_config_levels: int = DEFAULT_INDEX_LEVELS,
        index_config_clusters: int = DEFAULT_INDEX_CLUSTERS,
        index_tree_search_top_size: int = DEFAULT_INDEX_TREE_SEARCH_TOP_SIZE,
    ) -> None:
        """
        Args:
            host / port / database / secure: connection parameters (grpc or grpcs).
            credentials: None (anonymous), a dict (``{"username", "password"}``,
                ``{"token"}``, ``{"service_account_json"}``) or a ``ydb.Credentials``
                object; passed to ydb_dbapi as is.
            table_name: table name (letters, digits and underscores only).
            embed_dim: embedding dimension, checked on write and on search.
            perform_setup: Whether to create the table during initialization. If False, the table must already exist.
            batch_size: how many nodes to send in a single UPSERT.
            strategy: search metric (``YDBSearchStrategy`` or its string value).
                For an index it must match the metric the index was built with.
            index_enabled: read through the vector index ``index_name``. The index must
                exist (see ``rebuild_vector_index``), otherwise queries fail.
            index_name: name of the vector index.
            index_config_levels: number of tree levels (1-3 recommended).
            index_config_clusters: number of k-means clusters (64-512 recommended).
            index_tree_search_top_size: search completeness with the index
                (higher is more accurate and slower).

        """
        _validate_identifier(table_name, "table_name")
        _validate_identifier(index_name, "index_name")
        _positive_int(embed_dim, "embed_dim")
        _positive_int(batch_size, "batch_size")
        _positive_int(index_config_levels, "index_config_levels")
        _positive_int(index_config_clusters, "index_config_clusters")
        _positive_int(index_tree_search_top_size, "index_tree_search_top_size")
        try:
            strategy = YDBSearchStrategy(strategy)
        except ValueError:
            allowed = ", ".join(s.value for s in YDBSearchStrategy)
            raise ValueError(
                f"Unsupported strategy: {strategy!r}. Use one of: {allowed}"
            ) from None
        super().__init__(
            host=host,
            port=port,
            database=database,
            secure=secure,
            credentials=credentials,
            table_name=table_name,
            embed_dim=embed_dim,
            perform_setup=perform_setup,
            batch_size=batch_size,
            strategy=strategy,
            index_enabled=index_enabled,
            index_name=index_name,
            index_config_levels=index_config_levels,
            index_config_clusters=index_config_clusters,
            index_tree_search_top_size=index_tree_search_top_size,
        )

        self._aconnection_lock = asyncio.Lock()
        self._aindex_lock = asyncio.Lock()

        if self.perform_setup:
            self._setup()

    @classmethod
    def class_name(cls) -> str:
        return "YDBVectorStore"

    @classmethod
    def from_params(cls, **kwargs: Any) -> "YDBVectorStore":
        """Same parameters as the constructor (for consistency with other vector stores)."""
        return cls(**kwargs)

    @property
    def client(self) -> Any:
        """Sync ydb_dbapi connection (None until the store is first used)."""
        return self._connection

    def _create_table_query(self) -> str:
        """Build the CREATE TABLE statement for the nodes table."""
        return f"""
CREATE TABLE IF NOT EXISTS `{self.table_name}` (
    `node_id` Utf8,
    `ref_doc_id` Utf8,
    `text` Utf8,
    `metadata` Json,
    `embedding` String,
    PRIMARY KEY (`node_id`)
);
"""

    @staticmethod
    def _upsert_rows_type() -> ydb.ListType:
        """YDB type of the rows parameter of the UPSERT statement."""
        row = ydb.StructType()
        row.add_member("node_id", ydb.PrimitiveType.Utf8)
        row.add_member("ref_doc_id", ydb.OptionalType(ydb.PrimitiveType.Utf8))
        row.add_member("text", ydb.PrimitiveType.Utf8)
        row.add_member("metadata", ydb.PrimitiveType.Json)
        row.add_member("embedding", ydb.PrimitiveType.String)
        return ydb.ListType(row)

    @staticmethod
    def _ids_type() -> ydb.ListType:
        """YDB type of a list of node ids."""
        return ydb.ListType(ydb.PrimitiveType.Utf8)

    def _upsert_query(self) -> str:
        """Build the UPSERT statement that writes a batch of rows."""
        return f"""
DECLARE $rows AS List<Struct<
    node_id: Utf8,
    ref_doc_id: Utf8?,
    text: Utf8,
    metadata: Json,
    embedding: String>>;

UPSERT INTO `{self.table_name}` (`node_id`, `ref_doc_id`, `text`, `metadata`, `embedding`)
SELECT node_id, ref_doc_id, text, metadata, embedding
FROM AS_TABLE($rows);
"""

    def _search_query(self, top_k: int) -> str:
        """Search query; goes through the vector index when ``index_enabled`` is set."""
        _positive_int(top_k, "similarity_top_k")
        pragma = ""
        view = ""
        if self.index_enabled:
            pragma = f'PRAGMA ydb.KMeansTreeSearchTopSize="{self.index_tree_search_top_size}";\n'
            view = f" VIEW `{self.index_name}`"
        order = _strategy_sort_order(self.strategy)
        return f"""{pragma}
DECLARE $embedding AS String;

SELECT
    `node_id`,
    `text`,
    `metadata`,
    Knn::{self.strategy.value}(`embedding`, $embedding) AS score
FROM `{self.table_name}`{view}
ORDER BY score {order}
LIMIT {top_k};
"""

    def _delete_by_ref_doc_query(self) -> str:
        """Build the statement that deletes all nodes of one source document."""
        return f"""
DECLARE $ref_doc_id AS Utf8;

DELETE FROM `{self.table_name}`
WHERE `ref_doc_id` = $ref_doc_id;
"""

    def _get_by_ids_query(self) -> str:
        """Build the statement that selects nodes by id."""
        return f"""
DECLARE $ids AS List<Utf8>;

SELECT `node_id`, `text`, `metadata`, `embedding`
FROM `{self.table_name}`
WHERE `node_id` IN $ids;
"""

    def _delete_by_ids_query(self) -> str:
        """Build the statement that deletes nodes by id."""
        return f"""
DECLARE $ids AS List<Utf8>;

DELETE FROM `{self.table_name}`
WHERE `node_id` IN $ids;
"""

    def _clear_query(self) -> str:
        """Build the statement that deletes every row of the table."""
        return f"DELETE FROM `{self.table_name}`;"

    @property
    def _temp_index_name(self) -> str:
        """Name under which a new vector index is built before it replaces the old one."""
        return f"{self.index_name}__temp"

    def _add_vector_index_query(self) -> str:
        """DDL that builds the vector index under its temporary name."""
        return f"""
ALTER TABLE `{self.table_name}`
ADD INDEX `{self._temp_index_name}`
GLOBAL USING vector_kmeans_tree
ON (`embedding`)
WITH (
    {_strategy_index_param(self.strategy)},
    vector_type="Float",
    vector_dimension={self.embed_dim},
    levels={self.index_config_levels},
    clusters={self.index_config_clusters}
);
"""

    def _drop_temp_index_query(self) -> str:
        """Build the DDL that drops a leftover temporary index."""
        return f"ALTER TABLE `{self.table_name}` DROP INDEX `{self._temp_index_name}`;"

    def _rebuild_vector_index_query(self) -> str:
        """Build the YQL statement that rebuilds the existing vector index."""
        return f"""
    ALTER TABLE `{self.table_name}`
    REBUILD INDEX `{self.index_name}`
    WITH (
        clusters={self.index_config_clusters},
        levels={self.index_config_levels}
    );
    """

    def _build_metadata_filters(
        self,
        filters: MetadataFilters,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Metadata filters are not supported yet")

    def _search_params(self, embedding: Sequence[float]) -> dict:
        """Query parameters for the search statement."""
        return {"$embedding": (_vector_to_bytes(embedding), ydb.PrimitiveType.String)}

    def _delete_params(self, ref_doc_id: str) -> dict:
        """Query parameters for deleting a source document."""
        return {"$ref_doc_id": (ref_doc_id, ydb.PrimitiveType.Utf8)}

    def _ids_params(self, node_ids: Sequence[str]) -> tuple:
        """Query parameters for a list of node ids, plus the normalized id list."""
        id_list = [str(i) for i in node_ids]
        return id_list, {"$ids": (id_list, self._ids_type())}

    def _table_path(self) -> str:
        """Full path of the table, as expected by the YDB table client."""
        return f"{self.database.rstrip('/')}/{self.table_name}"

    def _prepare_add(self, nodes: Sequence[BaseNode]) -> List[Sequence[Dict[str, Any]]]:
        """Convert nodes to rows and split them into batches."""
        rows = [_node_to_row(node, self.embed_dim) for node in nodes]
        return list(_chunks(rows, self.batch_size))

    def _prepare_search(self, query: VectorStoreQuery) -> tuple:
        """Validate a query and return its YQL statement and parameters."""
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(
                f"YDBVectorStore supports only VectorStoreQueryMode.DEFAULT, got {query.mode}"
            )
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for a DEFAULT query")
        if len(query.query_embedding) != self.embed_dim:
            raise ValueError(
                f"query_embedding has {len(query.query_embedding)} dimensions, "
                f"but the store is configured with embed_dim={self.embed_dim}"
            )
        if query.filters is not None:
            raise NotImplementedError("Metadata filters are not supported yet")
        # NOTE: LlamaIndex passes node_ids=[] meaning "no restriction", so check truthiness
        if query.node_ids or query.doc_ids:
            raise NotImplementedError(
                "node_ids / doc_ids in queries are not supported yet"
            )
        return (
            self._search_query(query.similarity_top_k),
            self._search_params(query.query_embedding),
        )

    @staticmethod
    def _reject_filters(filters: Optional[MetadataFilters]) -> None:
        """Raise for metadata filters, which are not supported yet."""
        if filters is not None:
            raise NotImplementedError("Metadata filters are not supported yet")

    @staticmethod
    def _rows_as_dicts(cursor: Any) -> List[Dict[str, Any]]:
        """Convert the rows of a cursor to dictionaries keyed by column name."""
        if cursor.description is None:
            return []
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    @staticmethod
    def _run(
        connection: Any,
        query: str,
        params: Optional[dict] = None,
        ddl: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a statement on a sync connection and return its rows."""
        with connection.cursor() as cursor:
            if ddl:
                cursor.execute_scheme(query, params)
            else:
                cursor.execute(query, params)
            return YDBVectorStore._rows_as_dicts(cursor)

    @staticmethod
    async def _arun(
        connection: Any,
        query: str,
        params: Optional[dict] = None,
        ddl: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a statement on an async connection and return its rows."""
        async with connection.cursor() as cursor:
            if ddl:
                await cursor.execute_scheme(query, params)
            else:
                await cursor.execute(query, params)
            return YDBVectorStore._rows_as_dicts(cursor)

    def _connect_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ydb_dbapi.connect."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "credentials": self.credentials,
            "protocol": "grpcs" if self.secure else "grpc",
        }

    def _setup(self) -> None:
        """Create the table if it does not exist, without keeping a connection open."""
        connection = ydb_dbapi.connect(**self._connect_kwargs())
        try:
            self._run(connection, self._create_table_query(), ddl=True)
            self._table_ready = True
        finally:
            connection.close()

    def _get_connection(self) -> Any:
        """
        Return the sync connection.
        """
        if self._connection is None:
            self._connection = ydb_dbapi.connect(**self._connect_kwargs())
        return self._connection

    async def _aget_connection(self) -> Any:
        """Return the async connection bound to the current event loop."""
        loop = asyncio.get_running_loop()

        if self._aconnection is not None:
            if self._aloop is not loop:
                raise RuntimeError(
                    "YDBVectorStore async connection cannot be used "
                    "from a different event loop"
                )
            return self._aconnection

        async with self._aconnection_lock:
            if self._aconnection is None:
                connection = await ydb_dbapi.async_connect(**self._connect_kwargs())
                self._aconnection = connection
                self._aloop = loop

            return self._aconnection

    def _execute(
        self, query: str, params: Optional[dict] = None, ddl: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute a statement on the sync connection."""
        return self._run(self._get_connection(), query, params, ddl)

    async def _aexecute(
        self, query: str, params: Optional[dict] = None, ddl: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute a statement on the async connection."""
        return await self._arun(await self._aget_connection(), query, params, ddl)

    def _ensure_vector_index(self) -> bool:
        """
        Create the vector index lazily on the first indexed query.

        Index creation is intentionally deferred until query time so that the
        index is built after the initial data load, not on an empty table.

        Return result shows if index was created by this call.
        """
        index_created = False
        if not self.index_enabled or self._index_ready:
            return index_created

        with self._index_setup_lock:
            if self._index_ready:
                return index_created

            self._execute(self._add_vector_index_query(), ddl=True)

            driver = ydb.Driver(
                endpoint=f"{'grpcs' if self.secure else 'grpc'}://{self.host}:{self.port}",
                database=self.database,
                credentials=self.credentials,
            )
            try:
                driver.wait(timeout=10, fail_fast=True)
                driver.table_client.alter_table(
                    self._table_path(),
                    rename_indexes=[
                        _rename_item(self._temp_index_name, self.index_name)
                    ],
                )
                index_created = True
            finally:
                driver.stop()
            self._index_ready = True
            logger.info("Vector index %s created", self.index_name)
            return index_created

    async def _aensure_vector_index(self) -> bool:
        """Async variant of lazy vector-index creation."""
        index_created = False
        if not self.index_enabled or self._index_ready:
            return index_created

        async with self._aindex_lock:
            if self._index_ready:
                return index_created

            await self._aexecute(self._add_vector_index_query(), ddl=True)
            driver = ydb.aio.Driver(
                endpoint=f"{'grpcs' if self.secure else 'grpc'}://{self.host}:{self.port}",
                database=self.database,
                credentials=self.credentials,
            )
            try:
                await driver.wait(timeout=10, fail_fast=True)
                await driver.table_client.alter_table(
                    self._table_path(),
                    rename_indexes=[
                        _rename_item(self._temp_index_name, self.index_name)
                    ],
                )
                index_created = True
            finally:
                await driver.stop()
            self._index_ready = True
            logger.info("Vector index %s created", self.index_name)

            return index_created

    def close(self) -> None:
        """Close the sync connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def aclose(self) -> None:
        """Close the async connection (only if it belongs to the current loop)."""
        connection = self._aconnection
        if connection is None:
            return

        loop = asyncio.get_running_loop()
        if self._aloop is not loop:
            logger.debug(
                "Skipping async YDB connection close: "
                "connection belongs to a different event loop"
            )
            return

        await connection.close()
        self._aconnection = None
        self._aloop = None

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the vector store (existing ids are overwritten)."""
        if not nodes:
            return []
        batches = self._prepare_add(nodes)
        statement = self._upsert_query()
        rows_type = self._upsert_rows_type()
        for batch in batches:
            self._execute(statement, {"$rows": (list(batch), rows_type)})
        return [node.node_id for node in nodes]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store."""
        if self.index_enabled:
            self._ensure_vector_index()

        statement, params = self._prepare_search(query)
        return _rows_to_result(self._execute(statement, params), self.strategy)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes that belong to a source document."""
        self._execute(self._delete_by_ref_doc_query(), self._delete_params(ref_doc_id))

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id."""
        self._reject_filters(filters)
        if not node_ids:
            return
        _, params = self._ids_params(node_ids)

        query = self._delete_by_ids_query()
        if filters is not None:
            filter_query, filter_params = self._build_metadata_filters(filters)
            query += f" AND ({filter_query})"
            params.update(filter_params)

        self._execute(query, params)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes with their embeddings by id."""
        self._reject_filters(filters)
        if node_ids is None:
            raise ValueError("node_ids must be provided")
        if not node_ids:
            return []
        id_list, params = self._ids_params(node_ids)

        query = self._get_by_ids_query()
        if filters is not None:
            filter_query, filter_params = self._build_metadata_filters(filters)
            query += f" AND ({filter_query})"
            params.update(filter_params)

        rows = self._execute(query, params)
        return _rows_to_nodes(rows, id_list)

    def clear(self) -> None:
        """Delete all nodes from the table."""
        self._execute(self._clear_query())

    def rebuild_vector_index(self) -> None:
        """
        Create the index if missing, otherwise rebuild it.

        YDB builds a replacement index and atomically switches to it.
        The existing index remains available while the replacement is built.
        """
        if not self.index_enabled:
            raise ValueError("Cannot rebuild vector index when index_enabled=False")

        created = self._ensure_vector_index()

        if not created:
            self._execute(self._rebuild_vector_index_query(), ddl=True)

        logger.info("Vector index %s rebuilt", self.index_name)

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Add nodes to the vector store asynchronously."""
        if not nodes:
            return []
        batches = self._prepare_add(nodes)
        statement = self._upsert_query()
        rows_type = self._upsert_rows_type()
        for batch in batches:
            await self._aexecute(statement, {"$rows": (list(batch), rows_type)})
        return [node.node_id for node in nodes]

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query the vector store asynchronously."""
        if self.index_enabled:
            await self._aensure_vector_index()

        statement, params = self._prepare_search(query)
        return _rows_to_result(await self._aexecute(statement, params), self.strategy)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes that belong to a source document asynchronously."""
        await self._aexecute(
            self._delete_by_ref_doc_query(), self._delete_params(ref_doc_id)
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id asynchronously."""
        self._reject_filters(filters)
        if not node_ids:
            return
        _, params = self._ids_params(node_ids)
        query = self._delete_by_ids_query()
        if filters is not None:
            filter_query, filter_params = self._build_metadata_filters(filters)
            query += f" AND ({filter_query})"
            params.update(filter_params)

        await self._aexecute(query, params)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes with their embeddings by id asynchronously."""
        self._reject_filters(filters)
        if node_ids is None:
            raise ValueError("node_ids must be provided")
        if not node_ids:
            return []
        id_list, params = self._ids_params(node_ids)
        query = self._get_by_ids_query()
        if filters is not None:
            filter_query, filter_params = self._build_metadata_filters(filters)
            query += f" AND ({filter_query})"
            params.update(filter_params)

        rows = await self._aexecute(query, params)
        return _rows_to_nodes(rows, id_list)

    async def aclear(self) -> None:
        """Delete all nodes from the table asynchronously."""
        await self._aexecute(self._clear_query())

    async def arebuild_vector_index(self) -> None:
        """Async variant of rebuild_vector_index."""
        if not self.index_enabled:
            raise ValueError("Cannot rebuild vector index when index_enabled=False")

        created = await self._aensure_vector_index()

        if not created:
            await self._aexecute(self._rebuild_vector_index_query(), ddl=True)

        logger.info("Vector index %s rebuilt", self.index_name)
