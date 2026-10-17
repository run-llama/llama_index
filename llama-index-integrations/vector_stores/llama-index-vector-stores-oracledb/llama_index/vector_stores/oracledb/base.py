# OopCompanion:suppressRename
from __future__ import annotations

import array
import functools
import json
import logging
import math
import os
import re
import uuid
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
    Iterator,
)
from contextlib import contextmanager

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
    VectorStoreQueryMode,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
)

if TYPE_CHECKING:
    from oracledb import Connection

from llama_index.core.vector_stores.utils import metadata_dict_to_node
from pydantic import PrivateAttr

try:
    import oracledb
except ImportError as e:
    raise ImportError(
        "Unable to import oracledb, please install with `pip install -U oracledb`."
    ) from e


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


@contextmanager
def _get_connection(client: Any) -> Iterator[Connection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "ConnectionPool", None)

    if isinstance(client, oracledb.Connection):
        yield client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        with client.acquire() as connection:
            yield connection
    else:
        valid_types = "oracledb.Connection"
        if connection_pool_class:
            valid_types += " or oracledb.ConnectionPool"
        raise TypeError(
            f"Expected client of type {valid_types}, got {type(client).__name__}"
        )


def _escape_str(value: str) -> str:
    BS = "\\"
    must_escape = (BS, "'")
    return (
        "".join(f"{BS}{c}" if c in must_escape else c for c in value) if value else ""
    )


_IDENTIFIER_RE = re.compile(r'^(?:"[^"]+"|[^".]+)(?:\.(?:"[^"]+"|[^".]+))*$')
_IDENTIFIER_PART_RE = re.compile(r'"([^"]+)"|([^".]+)')


def _quote_identifier(name: str) -> str:
    parts = _identifier_parts(name)
    return ".".join(f'"{part}"' for part in parts)


def _identifier_parts(name: str) -> list[str]:
    if not isinstance(name, str):
        raise ValueError("Identifier name must be a string.")

    name = name.strip()
    if not name:
        raise ValueError("Identifier name must not be empty.")
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Identifier name {name} is not valid.")

    groups = _IDENTIFIER_PART_RE.findall(name)
    return [quoted or unquoted.upper() for quoted, unquoted in groups]


def _identifier_lookup_name(name: str) -> str:
    return _identifier_parts(name)[-1]


def _validate_int_param(
    config: dict[str, Any],
    key: str,
    min_value: int,
    max_value: Optional[int] = None,
) -> None:
    if key not in config:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < min_value:
        raise ValueError(f"{key} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} must be at most {max_value}.")


def _validate_index_type(config: dict[str, Any], expected_type: str) -> None:
    if "idx_type" not in config:
        return

    idx_type = config["idx_type"]
    if not isinstance(idx_type, str) or idx_type.upper() != expected_type:
        raise ValueError(f"idx_type must be {expected_type}.")
    config["idx_type"] = expected_type


def _validate_vector_index_common(config: dict[str, Any]) -> None:
    config["idx_name"] = _quote_identifier(config["idx_name"])
    _validate_int_param(config, "accuracy", 1, 100)
    _validate_int_param(config, "parallel", 1)


column_config: Dict = {
    "id": {"type": "VARCHAR2(64) PRIMARY KEY", "extract_func": lambda x: x.node_id},
    "doc_id": {"type": "VARCHAR2(64)", "extract_func": lambda x: x.ref_doc_id},
    "embedding": {
        "type": "VECTOR",
        "extract_func": lambda x: array.array("f", x.get_embedding()),
    },
    "node_info": {
        "type": "JSON",
        "extract_func": lambda x: json.dumps(x.get_node_info()),
    },
    "metadata": {
        "type": "JSON",
        "extract_func": lambda x: json.dumps(x.metadata),
    },
    "text": {
        "type": "CLOB",
        "extract_func": lambda x: _escape_str(
            x.get_content(metadata_mode=MetadataMode.NONE) or ""
        ),
    },
}


def _table_exists(connection: Connection, table_name: str) -> bool:
    try:
        import oracledb
    except ImportError as e:
        raise ImportError(
            "Unable to import oracledb, please install with `pip install -U oracledb`."
        ) from e
    try:
        table_name = _quote_identifier(table_name)
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


@_handle_exceptions
def _index_exists(connection: Connection, index_name: str) -> bool:
    # Check if the index exists
    parts = _identifier_parts(index_name)
    if len(parts) > 2:
        raise ValueError("Index name must be unqualified or schema-qualified.")

    query = "SELECT index_name FROM all_indexes WHERE index_name = :idx_name"
    params = {"idx_name": parts[-1]}
    if len(parts) == 2:
        query += " AND owner = :owner"
        params["owner"] = parts[0]

    with connection.cursor() as cursor:
        # Execute the query
        cursor.execute(query, **params)
        result = cursor.fetchone()

    # Check if the index exists
    return result is not None


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


def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f"{base_name}_{unique_id}"


@_handle_exceptions
def _create_table(connection: Connection, table_name: str) -> None:
    table_name = _quote_identifier(table_name)
    if not _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            column_definitions = ", ".join(
                [f"{k} {v['type']}" for k, v in column_config.items()]
            )

            # Generate the final DDL statement
            ddl = f"CREATE TABLE {table_name} (\n  {column_definitions}\n)"

            cursor.execute(ddl)
        logger.info("Table created successfully...")
    else:
        logger.info("Table already exists...")


@_handle_exceptions
def create_index(
    client: Any,
    vector_store: OraLlamaVS,
    params: Optional[dict[str, Any]] = None,
) -> None:
    with _get_connection(client) as connection:
        if params:
            params = params.copy()
            if "idx_name" in params:
                params["idx_name"] = _quote_identifier(params["idx_name"])
            idx_type = params.get("idx_type", "HNSW")
            if not isinstance(idx_type, str):
                raise ValueError("idx_type must be a string.")
            idx_type = idx_type.upper()
            params["idx_type"] = idx_type
            if idx_type == "HNSW":
                _create_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            elif idx_type == "IVF":
                _create_ivf_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            else:
                _create_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )


@_handle_exceptions
def _create_config(defaults: dict, params: Optional[dict]) -> dict:
    config: dict = {}
    if params:
        config = params.copy()
        # Ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(defaults[compulsory_key])
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # Validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults.copy()
    return config


@_handle_exceptions
def _create_hnsw_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    table_name = _quote_identifier(table_name)
    defaults = {
        "idx_name": "HNSW",
        "idx_type": "HNSW",
        "neighbors": 32,
        "efConstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    config = _create_config(defaults, params)
    if (
        "neighbors" in config or "efConstruction" in config
    ) and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "HNSW")
    _validate_int_param(config, "neighbors", 2, 2048)
    _validate_int_param(config, "efConstruction", 1, 65535)
    _validate_vector_index_common(config)

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = f"create vector index {idx_name} on {table_name}(embedding) ORGANIZATION INMEMORY NEIGHBOR GRAPH"

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if "accuracy" in config else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = " parameters (type {idx_type}, neighbors {neighbors}, efConstruction {efConstruction})"

    # Always included part for parallel
    parallel_part = " parallel {parallel}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not _index_exists(connection, config["idx_name"]):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
def _create_ivf_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    table_name = _quote_identifier(table_name)
    # Default configuration
    defaults = {
        "idx_name": "IVF",
        "idx_type": "IVF",
        "neighbor_part": 32,
        "accuracy": 90,
        "parallel": 8,
    }

    config = _create_config(defaults, params)
    if "neighbor_part" in config and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "IVF")
    _validate_int_param(config, "neighbor_part", 1, 10000000)
    _validate_vector_index_common(config)

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = f"CREATE VECTOR INDEX {idx_name} ON {table_name}(embedding) ORGANIZATION NEIGHBOR PARTITIONS"

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if "accuracy" in config else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = f" PARAMETERS (type {config['idx_type']}, neighbor partitions {config['neighbor_part']})"

    # Always included part for parallel
    parallel_part = f" PARALLEL {config['parallel']}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not _index_exists(connection, config["idx_name"]):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
def drop_table_purge(client: Any, table_name: str) -> None:
    table_name = _quote_identifier(table_name)
    with _get_connection(client) as connection:
        if _table_exists(connection, table_name):
            cursor = connection.cursor()
            with cursor:
                ddl = f"DROP TABLE {table_name} PURGE"
                cursor.execute(ddl)
            logger.info("Table dropped successfully...")
        else:
            logger.info("Table not found...")


@_handle_exceptions
def drop_index_if_exists(connection: Connection, index_name: str) -> None:
    index_name = _quote_identifier(index_name)
    if _index_exists(connection, index_name):
        drop_query = f"DROP INDEX {index_name}"
        with connection.cursor() as cursor:
            cursor.execute(drop_query)
            logger.info(f"Index {index_name} has been dropped.")
    else:
        logger.info(f"Index {index_name} does not exist.")


def output_type_string_handler(cursor: Any, metadata: Any) -> Any | None:
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)
    return None


def _generate_accum_query(query: str, fuzzy: Optional[bool] = False) -> str:
    """
    Tokenize query on non-word boundaries and join with Oracle Text ACCUM.

    Behavior:
    - Splits on non-word characters, discarding empty tokens.
    - When fuzzy is False: each token is quoted: "token".
    - When fuzzy is True: wraps each token as FUZZY("token").
    - Joins tokens with ' ACCUM '.

    Examples:
    'refund policy' -> '"refund" ACCUM "policy"'
    fuzzy=True -> 'fuzzy("refund") ACCUM fuzzy("policy")'

    """
    words = re.split(r"\W+", query)
    words = [f'"{word}"' if not fuzzy else f'fuzzy("{word}")' for word in words if word]
    return " ACCUM ".join(words)


class OraLlamaVS(BasePydanticVectorStore):
    """
    Oracle Database vector store for LlamaIndex.

    - Requires the ``oracledb`` Python package and an Oracle Database with Vector
      and Hybrid Search features enabled.
    - On initialization, creates the target table if it does not exist.
    - Supports:
      - DEFAULT mode: pure vector similarity using VECTOR_DISTANCE.
      - HYBRID mode: calls DBMS_HYBRID_VECTOR.SEARCH when ``VectorStoreQuery.mode == HYBRID``.
    - Hybrid search:
      - Create a hybrid vector index using ``create_hybrid_index`` from
        ``llama_index.vector_stores.oracledb.hybrid``.
      - Set ``hybrid_index_name`` (and optionally ``hybrid_search_params``) before
        querying in HYBRID mode.

    Example:
        .. code-block:: python

            import oracledb
            from llama_index.vector_stores.oracledb import OraLlamaVS, DistanceStrategy
            from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

            conn = oracledb.connect(dsn=os.environ["ORACLE_DB_DSN"])
            vs = OraLlamaVS(
                _client=conn,
                table_name="llama_index",
                distance_strategy=DistanceStrategy.COSINE,
            )

            # DEFAULT (vector) search
            q = VectorStoreQuery(
                query_str="hello",
                similarity_top_k=3,
                mode=VectorStoreQueryMode.DEFAULT,
            )
            res = vs.query(q)

    """

    AMPLIFY_RATIO_LE5: ClassVar[int] = 100
    AMPLIFY_RATIO_GT5: ClassVar[int] = 20
    AMPLIFY_RATIO_GT50: ClassVar[int] = 10
    metadata_column: str = "metadata"
    stores_text: bool = True
    _client: Connection = PrivateAttr()
    _quoted_table_name: str = PrivateAttr()
    table_name: str
    distance_strategy: DistanceStrategy
    batch_size: Optional[int]
    params: Optional[dict[str, Any]]
    hybrid_index_name: Optional[str] = None
    hybrid_search_params: Optional[dict] = None
    use_fuzzy_text_search: Optional[bool] = False

    def __init__(
        self,
        _client: Connection,
        table_name: str,
        distance_strategy: Optional[
            DistanceStrategy
        ] = DistanceStrategy.EUCLIDEAN_DISTANCE,
        batch_size: Optional[int] = 32,
        params: Optional[dict[str, Any]] = None,
        hybrid_index_name: Optional[str] = None,
        hybrid_search_params: Optional[dict] = None,
        use_fuzzy_text_search: Optional[bool] = False,
    ):
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        try:
            """Initialize with necessary components."""
            super().__init__(
                table_name=table_name,
                distance_strategy=distance_strategy,
                batch_size=batch_size,
                params=params,
            )
            with _get_connection(_client) as connection:
                # Assign _client to PrivateAttr after the Pydantic initialization
                object.__setattr__(self, "_client", _client)
                object.__setattr__(
                    self, "_quoted_table_name", _quote_identifier(table_name)
                )
                _create_table(connection, self._quoted_table_name)
            self.hybrid_index_name = hybrid_index_name
            self.hybrid_search_params = hybrid_search_params
            self.use_fuzzy_text_search = use_fuzzy_text_search

        except oracledb.DatabaseError as db_err:
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
        return "OraLlamaVS"

    def set_hybrid_index(self, hybrid_index_name: str) -> None:
        self.hybrid_index_name = hybrid_index_name

    def _convert_oper_to_sql(
        self,
        oper: FilterOperator,
        metadata_column: str,
        filter_key: str,
        value_bind: str,
    ) -> str:
        if oper == FilterOperator.IS_EMPTY:
            return f"NOT JSON_EXISTS({metadata_column}, '$.{filter_key}') OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), '[]') OR JSON_EQUAL(JSON_QUERY({metadata_column}, '$.{filter_key}'), 'null')"
        elif oper == FilterOperator.CONTAINS:
            return f"JSON_EXISTS({metadata_column}, '$.{filter_key}[*]?(@ == $val)' PASSING {value_bind} AS \"val\")"
        else:
            oper_map = {
                FilterOperator.EQ: "{0} = {1}",  # default operator (string, int, float)
                FilterOperator.GT: "{0} > {1}",  # greater than (int, float)
                FilterOperator.LT: "{0} < {1}",  # less than (int, float)
                FilterOperator.NE: "{0} != {1}",  # not equal to (string, int, float)
                FilterOperator.GTE: "{0} >= {1}",  # greater than or equal to (int, float)
                FilterOperator.LTE: "{0} <= {1}",  # less than or equal to (int, float)
                FilterOperator.IN: "{0} IN ({1})",  # In array (string or number)
                FilterOperator.NIN: "{0} NOT IN ({1})",  # Not in array (string or number)
                FilterOperator.TEXT_MATCH: "{0} LIKE '%' || {1} || '%'",  # full text match (allows you to search for a specific substring, token or phrase within the text field)
            }

            if oper not in oper_map:
                raise ValueError(
                    f"FilterOperation {oper} cannot be used with this vector store."
                )

            operation_f: str = oper_map.get(oper)
            returning_number = (
                "RETURNING NUMBER"
                if oper
                in [
                    FilterOperator.GT,
                    FilterOperator.LT,
                    FilterOperator.GTE,
                    FilterOperator.LTE,
                ]
                else ""
            )

            return operation_f.format(
                f"JSON_VALUE({metadata_column}, '$.{filter_key}' {returning_number})",
                value_bind,
            )

    def _get_filter_string(
        self, filter: MetadataFilters | MetadataFilter, bind_variables: list
    ) -> str:
        if isinstance(filter, MetadataFilter):
            if not re.match(r"^[a-zA-Z0-9_]+$", filter.key):
                raise ValueError(f"Invalid metadata key format: {filter.key}")

            value_bind = f""
            if filter.operator == FilterOperator.IS_EMPTY:
                # No values needed
                pass
            elif isinstance(filter.value, List):
                # Needs multiple binds for a list https://python-oracledb.readthedocs.io/en/latest/user_guide/bind.html#binding-multiple-values-to-a-sql-where-in-clause
                value_binds = []
                for val in filter.value:
                    value_binds.append(f":value{len(bind_variables)}")
                    bind_variables.append(val)
                value_bind = ",".join(value_binds)
            else:
                value_bind = f":value{len(bind_variables)}"
                bind_variables.append(filter.value)

            return self._convert_oper_to_sql(
                filter.operator, self.metadata_column, filter.key, value_bind
            )

        # Combine all sub filters
        filter_strings = [
            self._get_filter_string(f_, bind_variables) for f_ in filter.filters
        ]

        return f" {filter.condition.value.upper()} ".join(filter_strings)

    def _get_filter_json(self, filter: MetadataFilters | MetadataFilter) -> dict:
        if isinstance(filter, MetadataFilter):
            if not re.match(r"^[a-zA-Z0-9_]+$", filter.key):
                raise ValueError(f"Invalid metadata key format: {filter.key}")

            oper_map = {
                FilterOperator.EQ: "=",  # default operator (string, int, float)
                FilterOperator.GT: ">",  # greater than (int, float)
                FilterOperator.LT: "<",  # less than (int, float)
                FilterOperator.NE: "!=",  # not equal to (string, int, float)
                FilterOperator.GTE: ">=",  # greater than or equal to (int, float)
                FilterOperator.LTE: "<=",  # less than or equal to (int, float)
                FilterOperator.IN: "IN",  # In array (string or number)
                # FilterOperator.ANY: "ANY",  # Contains any (array of strings)
                # FilterOperator.ALL: "ALL",  # Contains all (array of strings)
                FilterOperator.TEXT_MATCH: "INSTR",  # full text match (allows you to search for a specific substring, token or phrase within the text field)
            }

            oper = filter.operator
            if oper not in oper_map:
                raise ValueError(
                    f"FilterOperation {oper} cannot be used with this vector store."
                )

            number_ops = {
                FilterOperator.GT,
                FilterOperator.LT,
                FilterOperator.GTE,
                FilterOperator.LTE,
            }

            string_ops = {
                FilterOperator.ANY,
                FilterOperator.ALL,
                FilterOperator.TEXT_MATCH,
            }

            op_type = "string"
            if oper in number_ops:
                op_type = "number"
            elif oper in string_ops:
                op_type = "string"
            elif isinstance(filter.value, (int, float)):
                op_type = "number"
            elif isinstance(filter.value, str):
                op_type = "string"
            elif isinstance(filter.value, list):
                if isinstance(filter.value[0], (int, float)):
                    op_type = "number"
                elif isinstance(filter.value[0], str):
                    op_type = "string"

            return {
                "op": oper_map[oper],
                "path": f"metadata.{filter.key}",
                "type": op_type,
                "args": filter.value
                if isinstance(filter.value, list)
                else [filter.value],
            }

        # Combine all sub filters
        filter_strings = [self._get_filter_json(f_) for f_ in filter.filters]

        if len(filter_strings) == 1:
            return filter_strings[0]

        return {"op": filter.condition.value.upper(), "args": filter_strings}

    def _append_meta_filter_condition(
        self, where_str: Optional[str], filters: Optional[MetadataFilters]
    ) -> Tuple[str, list]:
        bind_variables: list = []

        filter_str = self._get_filter_string(filters, bind_variables)

        # Convert filter conditions to a single string
        if where_str is None:
            where_str = filter_str
        else:
            where_str += " AND " + filter_str

        return where_str, bind_variables

    def _build_insert(self, values: List[BaseNode]) -> Tuple[str, List[tuple]]:
        _data = []
        for item in values:
            item_values = tuple(
                column["extract_func"](item) for column in column_config.values()
            )
            _data.append(item_values)

        dml = f"""
           INSERT INTO {self._quoted_table_name} ({", ".join(column_config.keys())})
           VALUES ({", ".join([":" + str(i + 1) for i in range(len(column_config))])})
        """
        return dml, _data

    def _build_query(
        self, distance_function: str, k: int, where_str: Optional[str] = None
    ) -> str:
        where_clause = f"WHERE {where_str}" if where_str else ""

        return f"""
            SELECT id,
                doc_id,
                text,
                node_info,
                metadata,
                vector_distance(embedding, :embedding, {distance_function}) AS distance
            FROM {self._quoted_table_name}
            {where_clause}
            ORDER BY distance
            FETCH APPROX FIRST {k} ROWS ONLY
        """

    def _get_search_params(self, query: VectorStoreQuery, **kwargs: Any) -> dict:
        """
        Build the JSON-serializable parameter dict for DBMS_HYBRID_VECTOR.SEARCH.

        Behavior:
        - Sets "hybrid_index_name" from ``self.hybrid_index_name``.
        - Sets both "vector.search_text" and "text.search_text" to the query string.
        - Sets "return.topN", "return.values" and "return.format" ("JSON").

        Args:
            query: VectorStoreQuery carrying the user query text and top-k.

        Returns:
            dict: Parameters suitable for binding as json(:search_params)
                in DBMS_HYBRID_VECTOR.SEARCH.

        """
        query_text = query.query_str
        search_params = dict(self.hybrid_search_params or {})

        if not self.hybrid_index_name:
            raise ValueError("Need to set `hybrid_index_name`")
        search_params["hybrid_index_name"] = _quote_identifier(self.hybrid_index_name)

        if "search_text" in search_params:
            raise ValueError(
                "Cannot provide search_text as a parameter at the top level; "
                "it is derived from the query."
            )
        if "return" in search_params:
            raise ValueError(
                "Cannot provide return as a parameter in params; "
                "it is handled internally. Use `return_scores` "
                "parameter to get the scores."
            )

        search_params["vector"] = dict(search_params.get("vector") or {})

        if (
            "search_text" in search_params["vector"]
            or "search_vector" in search_params["vector"]
        ):
            raise ValueError(
                "Cannot provide search_text as a parameter in params['vector']; "
                "it is derived from the query."
            )

        search_params["vector"]["search_text"] = query_text

        search_params["text"] = dict(search_params.get("text") or {})
        if (
            "search_text" in search_params["text"]
            or "search_vector" in search_params["text"]
            or "contains" in search_params["text"]
        ):
            raise ValueError(
                "Cannot provide search_text as a parameter in params['text']; "
                "it is derived from the query."
            )
        search_params["text"]["search_text"] = query_text

        search_params["return"] = {}
        search_params["return"]["topN"] = query.hybrid_top_k or 4
        search_params["return"]["values"] = [
            "rowid",
            "score",
            "vector_score",
            "text_score",
        ]
        search_params["return"]["format"] = "JSON"

        return search_params

    @_handle_exceptions
    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []

        with _get_connection(self._client) as connection:
            for result_batch in iter_batch(nodes, self.batch_size):
                dml, bind_values = self._build_insert(values=result_batch)

                with connection.cursor() as cursor:
                    # Use executemany to insert the batch
                    cursor.executemany(dml, bind_values)
                    connection.commit()

        return [node.node_id for node in nodes]

    @_handle_exceptions
    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                ddl = (
                    f"DELETE FROM {self._quoted_table_name} WHERE doc_id = :ref_doc_id"
                )
                cursor.execute(ddl, [ref_doc_id])
                connection.commit()

    @_handle_exceptions
    def _get_clob_value(self, result: Any) -> str:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        clob_value = ""
        if result:
            if isinstance(result, oracledb.LOB):
                raw_data = result.read()
                if isinstance(raw_data, bytes):
                    clob_value = raw_data.decode(
                        "utf-8"
                    )  # Specify the correct encoding
                else:
                    clob_value = raw_data
            elif isinstance(result, str):
                clob_value = result
            else:
                raise Exception("Unexpected type:", type(result))
        return clob_value

    @_handle_exceptions
    def drop(self) -> None:
        drop_table_purge(self._client, self.table_name)

    def _get_default_query(self, query: VectorStoreQuery) -> Tuple[str, Dict[str, Any]]:
        distance_function = _get_distance_function(self.distance_strategy)
        where_str = None
        params = {}

        if query.doc_ids:
            placeholders = ", ".join([f":doc_id{i}" for i in range(len(query.doc_ids))])
            where_str = f"doc_id in ({placeholders})"
            for i, doc_id in enumerate(query.doc_ids):
                params[f"doc_id{i}"] = doc_id

        bind_vars = []
        if query.filters is not None:
            where_str, bind_vars = self._append_meta_filter_condition(
                where_str, query.filters
            )

        # build query sql
        query_sql = self._build_query(
            distance_function, query.similarity_top_k, where_str
        )

        embedding = array.array("f", query.query_embedding)
        params["embedding"] = embedding
        for i, value in enumerate(bind_vars):
            params[f"value{i}"] = value

        return query_sql, params

    def _get_hybrid_query(self, query: VectorStoreQuery) -> Tuple[str, Dict[str, Any]]:
        SQL_QUERY = "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))"

        json_filter = {}
        if query.doc_ids:
            json_filter = {
                "op": "IN",
                "col": "doc_id",
                "type": "string",
                "args": query.doc_ids,
            }

        if query.filters is not None:
            json_filter_metadata = self._get_filter_json(query.filters)
            if len(json_filter) > 0:
                json_filter = {"op": "AND", "args": [json_filter, json_filter_metadata]}
            else:
                json_filter = json_filter_metadata

        search_params = self._get_search_params(query)

        if len(json_filter) > 0:
            search_params["filter_by"] = json_filter

        return SQL_QUERY, {"search_params": search_params}

    def _get_text_query(self, query: VectorStoreQuery) -> Tuple[str, Dict[str, Any]]:
        where_str = None
        params = {}

        if query.doc_ids:
            placeholders = ", ".join([f":doc_id{i}" for i in range(len(query.doc_ids))])
            where_str = f"doc_id in ({placeholders})"
            for i, doc_id in enumerate(query.doc_ids):
                params[f"doc_id{i}"] = doc_id

        bind_vars = []
        if query.filters is not None:
            where_str, bind_vars = self._append_meta_filter_condition(
                where_str, query.filters
            )

        text_query = _generate_accum_query(query.query_str, self.use_fuzzy_text_search)
        k = query.similarity_top_k or 4

        params["query"] = text_query

        for i, value in enumerate(bind_vars):
            params[f"value{i}"] = value

        search_query = f"""
        SELECT id,
                doc_id,
                text,
                node_info,
                metadata,
                SCORE(1) score
        FROM {self._quoted_table_name}
        WHERE CONTAINS(text, :query, 1) > 0
        {"AND " + where_str if where_str else ""}
        ORDER BY score DESC FETCH FIRST {k} ROWS ONLY
        """

        return search_query, params

    def _resolve_rowids(self, result: Any) -> list:
        rowids = []
        scores = []
        text_scores = []
        vector_scores = []
        for row in result:
            rowids.append((row["rowid"],))
            scores.append(row["score"])
            text_scores.append(row["text_score"])
            vector_scores.append(row["vector_score"])

        res = []
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                for i, rid_tuple in enumerate(rowids):
                    rid = rid_tuple[0]
                    cursor.execute(
                        "SELECT id, doc_id, text, node_info, metadata "
                        f"FROM {self._quoted_table_name} "
                        "WHERE rowid = :1",
                        [rid],
                    )
                    row = cursor.fetchone()
                    res.append((*row, scores[i]))

        return res

    @_handle_exceptions
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode == VectorStoreQueryMode.DEFAULT:
            query_sql, params = self._get_default_query(query)

        if query.mode == VectorStoreQueryMode.HYBRID:
            query_sql, params = self._get_hybrid_query(query)

        if query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            query_sql, params = self._get_text_query(query)

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                if query.mode == VectorStoreQueryMode.HYBRID:
                    cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                cursor.execute(query_sql, **params)
                results = cursor.fetchall()

                if query.mode == VectorStoreQueryMode.HYBRID:
                    results = results[0][0]
                    if hasattr(results, "read"):
                        results = results.read()
                    results = self._resolve_rowids(json.loads(results))

                similarities = []
                ids = []
                nodes = []
                for result in results:
                    doc_id = result[1]
                    text = self._get_clob_value(result[2])
                    node_info = (
                        json.loads(result[3])
                        if isinstance(result[3], str)
                        else result[3]
                    )
                    metadata = (
                        json.loads(result[4])
                        if isinstance(result[4], str)
                        else result[4]
                    )

                    if query.node_ids:
                        if result[0] not in query.node_ids:
                            continue

                    start_char_idx = None
                    end_char_idx = None
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
                    if (
                        query.mode == VectorStoreQueryMode.HYBRID
                        or query.mode == VectorStoreQueryMode.TEXT_SEARCH
                    ):
                        similarities.append(result[5])
                    else:
                        similarities.append(1.0 - math.exp(-result[5]))
                    ids.append(result[0])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @classmethod
    @_handle_exceptions
    def from_documents(
        cls: Type[OraLlamaVS],
        docs: List[TextNode],
        table_name: str = "llama_index",
        **kwargs: Any,
    ) -> OraLlamaVS:
        """Return VectorStore initialized from texts and embeddings."""
        _client = kwargs.get("client")
        if _client is None:
            raise ValueError("client parameter is required...")
        params = kwargs.get("params")
        distance_strategy = kwargs.get("distance_strategy")
        drop_table_purge(_client, table_name)

        vss = cls(
            _client=_client,
            table_name=table_name,
            params=params,
            distance_strategy=distance_strategy,
        )
        vss.add(nodes=docs)
        return vss
