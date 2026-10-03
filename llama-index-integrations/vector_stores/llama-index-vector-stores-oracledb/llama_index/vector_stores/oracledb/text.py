"""
Full-text search utilities for Oracle Database (Oracle Text).

This module provides:
- create_text_index/acreate_text_index: build an Oracle Text SEARCH INDEX
  over a table column (either using an OracleVS table or a user-provided table).
- OracleTextSearchRetriever: a LangChain retriever that executes Oracle Text
  CONTAINS queries and returns LangChain Documents.

Notes:
- When a vector_store (OracleVS) is provided, the supported searchable columns are
  limited to "text".
- You may also target an arbitrary table/column by supplying
  (client + table_name + column_name).

Query tips:
- operator_search=False (default): the input is treated as literal text. It is
  tokenized on non-word characters and rewritten as an ACCUM expression of the
  tokens. Each token is quoted, or when fuzzy=True, wrapped as FUZZY("token").

Examples:
    "refund policy" -> '"refund" ACCUM "policy"'
    fuzzy=True      -> 'fuzzy("refund") ACCUM fuzzy("policy")'
- operator_search=True: the input is treated as an Oracle Text expression and
  sent to CONTAINS unchanged (operators like NEAR, ABOUT, AND, OR, NOT, WITHIN,
  etc. are honored). In this mode, fuzzy is ignored.
- fuzzy helps match misspellings when operator_search=False by applying Oracle
  Text FUZZY per token. See:
  https://docs.oracle.com/en/database/oracle/oracle-database/26/ccref/oracle-text-CONTAINS-query-operators.html
- Results are ordered by score descending; use return_scores=True to include
  the score in each Document's metadata as "score".

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union


from llama_index.vector_stores.oracledb import OraLlamaVS
from llama_index.vector_stores.oracledb.base import (
    _get_connection,
    _handle_exceptions,
    _index_exists,
    _quote_identifier,
)

if TYPE_CHECKING:
    from oracledb import (
        Connection,
        ConnectionPool,
    )

logger = logging.getLogger(__name__)


def _get_text_index_ddl(
    idx_name: str,
    vector_store: OraLlamaVS,
) -> str:
    """
    Build the CREATE SEARCH INDEX DDL statement and resolve the target table.

    Args:
        idx_name: Index name (will be quoted).
        vector_store: OracleVS instance. If provided, the table name is taken from it,
            and column_name must be "text".
        table_name: Explicit table name (quoted). Mutually exclusive with vector_store.
        column_name: Column to index. Defaults to "text". When vector_store is given,
            allowed value is only "text".

    Returns:
        tuple[str, str]: (ddl, resolved_table_name)

    Raises:
        ValueError: If both vector_store and table_name are provided, or if neither
            resolves to a valid target; also for invalid column choices.

    """
    idx_name = _quote_identifier(idx_name)
    table_name = vector_store._quoted_table_name
    col = "text"

    return f"CREATE SEARCH INDEX {idx_name} ON {table_name}({col})"


@_handle_exceptions
def create_text_index(
    client: Union["Connection", "ConnectionPool"],
    idx_name: str,
    vector_store: OraLlamaVS,
) -> None:
    """
    Create an Oracle Text SEARCH INDEX if it does not already exist.

    Exactly one of vector_store or table_name must be provided.
    - If vector_store is given, column_name must be "text".
    - If table_name is given, column_name is required and used as-is (unquoted).

    Args:
        client: oracledb connection or connection pool.
        idx_name: Index name to create (quoted automatically).
        vector_store: OracleVS backing table to index ("text").
        table_name: Explicit table to index.
        column_name: Column to index. Defaults to "text".

    Raises:
        RuntimeError/ValueError: on DB/validation errors.

    """
    ddl = _get_text_index_ddl(idx_name, vector_store)

    with _get_connection(client) as connection:
        if not _index_exists(connection, idx_name):
            with connection.cursor() as cur:
                cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")
