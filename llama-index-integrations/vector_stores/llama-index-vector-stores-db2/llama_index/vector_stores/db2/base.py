# OopCompanion:suppressRename
from __future__ import annotations

import functools
import json
import logging
import math
import os
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

if TYPE_CHECKING:
    from ibm_db_dbi import Connection

from llama_index.core.vector_stores.utils import metadata_dict_to_node
from pydantic import PrivateAttr


logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DistanceStrategy(Enum):
    COSINE = 1
    DOT_PRODUCT = 2
    EUCLIDEAN_DISTANCE = 3
    MANHATTAN_DISTANCE = 4
    HAMMING_DISTANCE = 5
    EUCLIDEAN_SQUARED = 6


# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(f"Failed due to a DB issue: {db_err}") from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError(f"Validation failed: {val_err}") from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected error: {e}") from e

    return cast(T, wrapper)


def _escape_str(value: str) -> str:
    BS = "\\"
    must_escape = (BS, "'")
    return (
        "".join(f"{BS}{c}" if c in must_escape else c for c in value) if value else ""
    )


column_config: Dict = {
    "id": {"type": "VARCHAR(64) PRIMARY KEY", "extract_func": lambda x: x.node_id},
    "doc_id": {"type": "VARCHAR(64)", "extract_func": lambda x: x.ref_doc_id},
    "embedding": {
        "type": "VECTOR(embedding_dim, FLOAT32)",
        "extract_func": lambda x: f"{x.get_embedding()}",
    },
    "node_info": {
        "type": "BLOB",
        "extract_func": lambda x: json.dumps(x.node_info),
    },
    "metadata": {
        "type": "BLOB",
        "extract_func": lambda x: json.dumps(x.metadata),
    },
    "text": {
        "type": "CLOB",
        "extract_func": lambda x: _escape_str(
            x.get_content(metadata_mode=MetadataMode.NONE) or ""
        ),
    },
}


def _stringify_list(lst: List) -> str:
    return "(" + ",".join(f"'{item}'" for item in lst) + ")"


def table_exists(connection: Connection, table_name: str) -> bool:
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    except Exception as ex:
        if "SQL0204N" in str(ex):
            return False
        raise
    finally:
        cursor.close()
    return True


def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # Dictionary to map distance strategies to their corresponding function names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
        DistanceStrategy.MANHATTAN_DISTANCE: "MANHATTAN",
        DistanceStrategy.HAMMING_DISTANCE: "HAMMING",
        DistanceStrategy.EUCLIDEAN_SQUARED: "EUCLIDEAN_SQUARED",
    }

    # Attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # If it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")


@_handle_exceptions
def create_table(client: Connection, table_name: str, embedding_dim: int) -> None:
    cols_dict = {
        "id": "VARCHAR(64) PRIMARY KEY NOT NULL",
        "doc_id": "VARCHAR(64)",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
        "node_info": "BLOB",
        "metadata": "BLOB",
        "text": "CLOB",
    }

    if not table_exists(client, table_name):
        cursor = client.cursor()
        ddl_body = ", ".join(
            f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
        )
        ddl = f"CREATE TABLE {table_name} ({ddl_body})"
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
            logger.info(f"Table {table_name} created successfully...")
        finally:
            cursor.close()
    else:
        logger.info(f"Table {table_name} already exists...")


@_handle_exceptions
def drop_table(connection: Connection, table_name: str) -> None:
    if table_exists(connection, table_name):
        cursor = connection.cursor()
        try:
            ddl = f"DROP TABLE {table_name}"
            cursor.execute(ddl)
            logger.info("Table dropped successfully...")
        finally:
            cursor.close()
    else:
        logger.info("Table not found...")


class DB2LlamaVS(BasePydanticVectorStore):
    """
    `DB2LlamaVS` vector store.

    To use, you should have both:
    - the ``ibm_db`` python package installed
    - a connection to db2 database with vector store feature (v12.1.2+)
    """

    metadata_column: str = "metadata"
    stores_text: bool = True
    _client: Connection = PrivateAttr()
    table_name: str
    distance_strategy: DistanceStrategy
    batch_size: Optional[int]
    params: Optional[dict[str, Any]]
    embed_dim: int

    def __init__(
        self,
        _client: Connection,
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        batch_size: Optional[int] = 32,
        embed_dim: int = 1536,
        params: Optional[dict[str, Any]] = None,
    ):
        try:
            import ibm_db_dbi
        except ImportError as e:
            raise ImportError(
                "Unable to import ibm_db_dbi, please install with "
                "`pip install -U ibm_db`."
            ) from e

        try:
            """Initialize with necessary components."""
            super().__init__(
                table_name=table_name,
                distance_strategy=distance_strategy,
                batch_size=batch_size,
                embed_dim=embed_dim,
                params=params,
            )
            # Assign _client to PrivateAttr after the Pydantic initialization
            object.__setattr__(self, "_client", _client)
            create_table(_client, table_name, embed_dim)

        except ibm_db_dbi.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the index.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "DB2LlamaVS"

    def _append_meta_filter_condition(
        self, where_str: Optional[str], exact_match_filter: list
    ) -> str:
        filter_str = " AND ".join(
            f"JSON_VALUE({self.metadata_column}, '$.{filter_item.key}') = '{filter_item.value}'"
            for filter_item in exact_match_filter
        )
        if where_str is None:
            where_str = filter_str
        else:
            where_str += " AND " + filter_str
        return where_str

    def _build_insert(self, values: List[BaseNode]) -> List[tuple]:
        _data = []
        for item in values:
            item_values = tuple(
                column["extract_func"](item) for column in column_config.values()
            )
            _data.append(item_values)

        return _data

    def _build_query(
        self, distance_function: str, k: int, where_str: Optional[str] = None
    ) -> str:
        where_clause = f"WHERE {where_str}" if where_str else ""

        return f"""
            SELECT id,
                doc_id,
                text,
                SYSTOOLS.BSON2JSON(node_info),
                SYSTOOLS.BSON2JSON(metadata),
                vector_distance(embedding, VECTOR(?, {self.embed_dim}, FLOAT32), {distance_function}) AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance
            FETCH FIRST {k} ROWS ONLY
        """

    @_handle_exceptions
    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []

        for result_batch in iter_batch(nodes, self.batch_size):
            bind_values = self._build_insert(values=result_batch)

        dml = f"""
           INSERT INTO {self.table_name} ({", ".join(column_config.keys())})
           VALUES (?, ?, VECTOR(?, {self.embed_dim}, FLOAT32), SYSTOOLS.JSON2BSON(?), SYSTOOLS.JSON2BSON(?), ?)
        """

        cursor = self.client.cursor()
        try:
            # Use executemany to insert the batch
            cursor.executemany(dml, bind_values)
            cursor.execute("COMMIT")
        finally:
            cursor.close()

        return [node.node_id for node in nodes]

    @_handle_exceptions
    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        ddl = f"DELETE FROM {self.table_name} WHERE doc_id = '{ref_doc_id}'"
        cursor = self._client.cursor()
        try:
            cursor.execute(ddl)
            cursor.execute("COMMIT")
        finally:
            cursor.close()

    @_handle_exceptions
    def drop(self) -> None:
        drop_table(self._client, self.table_name)

    @_handle_exceptions
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        distance_function = _get_distance_function(self.distance_strategy)
        where_str = (
            f"doc_id in {_stringify_list(query.doc_ids)}" if query.doc_ids else None
        )

        if query.filters is not None:
            where_str = self._append_meta_filter_condition(
                where_str, query.filters.filters
            )

        # build query sql
        query_sql = self._build_query(
            distance_function, query.similarity_top_k, where_str
        )

        embedding = f"{query.query_embedding}"
        cursor = self._client.cursor()
        try:
            cursor.execute(query_sql, [embedding])
            results = cursor.fetchall()
        finally:
            cursor.close()

        similarities = []
        ids = []
        nodes = []
        for result in results:
            doc_id = result[1]
            text = result[2] if result[2] is not None else ""
            node_info = json.loads(result[3] if result[3] is not None else "{}")
            metadata = json.loads(result[4] if result[4] is not None else "{}")

            if query.node_ids:
                if result[0] not in query.node_ids:
                    continue

            if isinstance(node_info, dict):
                start_char_idx = node_info.get("start", None)
                end_char_idx = node_info.get("end", None)
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(text)
            except Exception:
                # Note: deprecated legacy logic for backward compatibility

                node = TextNode(
                    id_=result[0],
                    text=text,
                    metadata=metadata,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                    },
                )

            nodes.append(node)
            similarities.append(1.0 - math.exp(-result[5]))
            ids.append(result[0])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @classmethod
    @_handle_exceptions
    def from_documents(
        cls: Type[DB2LlamaVS],
        docs: List[BaseNode],
        table_name: str = "llama_index",
        **kwargs: Any,
    ) -> DB2LlamaVS:
        """Return VectorStore initialized from texts and embeddings."""
        _client = kwargs.get("client")
        if _client is None:
            raise ValueError("client parameter is required...")
        params = kwargs.get("params")
        distance_strategy = kwargs.get("distance_strategy")
        drop_table(_client, table_name)
        embed_dim = kwargs.get("embed_dim")

        vss = cls(
            _client=_client,
            table_name=table_name,
            params=params,
            distance_strategy=distance_strategy,
            embed_dim=embed_dim,
        )
        vss.add(nodes=docs)
        return vss
