"""
Hybrid search utilities for Oracle Database.

This module provides:
- OracleVectorizerPreference: manage DBMS_VECTOR_CHAIN vectorizer preferences used by
  hybrid vector indexes.
- create_hybrid_index: create a HYBRID VECTOR INDEX over the text column using a
  vectorizer preference.
- Integration with LlamaIndex: OraLlamaVS can execute hybrid retrieval by calling
  DBMS_HYBRID_VECTOR.SEARCH when VectorStoreQuery.mode == HYBRID. Set
  OraLlamaVS.hybrid_index_name (and optionally OraLlamaVS.hybrid_search_params)
  before issuing a hybrid query.

References:
- Hybrid search overview and guidance: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/understand-hybrid-search.html#GUID-310D2298-90F4-4AFE-AF03-F3B81E55F84C__GUID-03905981-A6E9-4D2C-A0DC-0807A95AA3F3
- Create vectorizer preference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html
- Create hybrid vector index: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
- Search API (DBMS_HYBRID_VECTOR.SEARCH): https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html

"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from llama_index.embeddings.oracleai import OracleEmbeddings
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


def _quote_filter_order_identifier(value: str, field_name: str) -> str:
    value = value.strip()
    simple_identifier = r"[A-Za-z][A-Za-z0-9_$#]*"
    reg = (
        rf'^(?:"{simple_identifier}"|{simple_identifier})'
        rf'(?:\.(?:"{simple_identifier}"|{simple_identifier}))*$'
    )
    if not re.fullmatch(reg, value):
        raise ValueError(f"{field_name} contains an invalid identifier")

    pattern_match = rf'"({simple_identifier})"|({simple_identifier})'
    groups = re.findall(pattern_match, value)
    quoted_groups = [
        f'"{quoted}"' if quoted else f'"{unquoted.upper()}"'
        for quoted, unquoted in groups
    ]
    return ".".join(quoted_groups)


def _quote_identifier_list(values: Any, field_name: str) -> str:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} must contain only column names")
    return ",".join(
        _quote_filter_order_identifier(value, field_name) for value in values
    )


def _validate_parameters(
    embeddings: OracleEmbeddings,
    params: dict[str, Any],
) -> bool:
    """
    Validate that provided preference parameters are consistent with the
    OracleEmbeddings bound to the vector store.

    Supports two mutually exclusive ways to specify the model configuration:
    - model: database-resident embedding model name
    - embedder_spec: JSON spec for external embedding providers

    Raises:
        ValueError: if parameters do not match the embeddings configuration.

    Returns:
        bool: True if a model configuration was explicitly provided; False otherwise.

    """
    if "model" in params:
        model_name = params.get("model")

        if (
            embeddings._params.get("provider") != "database"
            or embeddings._params.get("model") != model_name
        ):
            raise ValueError(
                f"Mismatch between embedding and provided params: "
                f"OracleEmbeddings expects provider='database' and "
                f"model='{embeddings._params.get('model')}', but received "
                f"model='{model_name}'."
            )

        return True

    if "embedder_spec" in params:
        embedder_spec = params.get("embedder_spec")

        if not (
            json.dumps(embeddings._params, sort_keys=True)
            == json.dumps(embedder_spec, sort_keys=True)
        ):
            raise ValueError(
                "Mismatch between embedding and provided params: "
                "embedder_spec must exactly match OracleEmbeddings._params "
                "(after JSON normalization)."
            )

        return True

    return False


class OracleVectorizerPreference:
    """Manage DBMS_VECTOR_CHAIN vectorizer preferences for hybrid search.

    A vectorizer preference encapsulates embedding configuration used by
    hybrid vector indexes. This class derives the correct preference parameters
    from the provided OracleEmbeddings.

    Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html

    """  # noqa E501

    params: Optional[dict[str, Any]]
    preference_name: str
    embeddings: OracleEmbeddings
    client: Any

    PREFERENCE_STR = """
    begin
    DBMS_VECTOR_CHAIN.CREATE_PREFERENCE(
        :1,
        dbms_vector_chain.vectorizer,
        json(:2));
    end;"""

    def _get_preference_parameters(self) -> dict:
        preference_params = self.params.copy() if self.params else {}
        embeddings = self.embeddings
        if not isinstance(embeddings, OracleEmbeddings):
            raise ValueError(
                "Only OracleEmbeddings can be used to create a vectorizer preference; "
                f"received type {type(embeddings).__name__}."
            )

        has_model_config = _validate_parameters(embeddings, preference_params)

        if not has_model_config:
            if embeddings._params.get("provider") == "database":
                preference_params["model"] = embeddings._params.get("model")
            else:
                preference_params["embedder_spec"] = embeddings._params

        return preference_params

    @classmethod
    def create_preference(
        cls,
        client: Any,
        embeddings: OracleEmbeddings,
        preference_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> OracleVectorizerPreference:
        """
        Create a DBMS_VECTOR_CHAIN vectorizer preference for hybrid indexing.

        Parameters are derived from the provided OracleEmbeddings unless explicitly
        overridden via params.

        Args:
            client: oracledb.Connection or oracledb.ConnectionPool used to execute DDL.
            embeddings: OracleEmbeddings whose configuration defines the vectorizer.
            preference_name: Optional explicit preference name. A random name is
                generated if omitted.
            params: Optional dict with additional options. You may either provide:
                - model: database-resident embedding model name, or
                - embedder_spec: JSON spec for an external embedding provider.
              If omitted, values are inferred from `embeddings`.

        Returns:
            OracleVectorizerPreference: handle containing the created preference name.

        Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html

        """
        self = cls.__new__(cls)
        self.params = params
        self.embeddings = embeddings
        self.preference_name = (
            preference_name or "pref" + str(uuid.uuid4()).replace("-", "")[0:15]
        )
        self.client = client

        preference_params = self._get_preference_parameters()

        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    OracleVectorizerPreference.PREFERENCE_STR,
                    [self.preference_name, json.dumps(preference_params)],
                )
                logger.info(f"Preference {self.preference_name} created.")

        return self

    def drop_preference(self) -> None:
        """Drop this vectorizer preference using DBMS_VECTOR_CHAIN.DROP_PREFERENCE."""
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                    preference_name=self.preference_name,
                )


def drop_preference(connection: Any, preference_name: str) -> None:
    """
    Drop a DBMS_VECTOR_CHAIN preference by name.

    Args:
        connection: oracledb connection.
        preference_name: Preference to drop.

    """
    with _get_connection(connection) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                preference_name=preference_name,
            )


def _get_hybrid_index_ddl(
    vectorizer_preference: OracleVectorizerPreference,
    idx_name: str,
    table_name: str,
    params: dict[str, Any],
) -> str:
    """
    Build the CREATE HYBRID VECTOR INDEX DDL statement.

    The vectorizer is set via the provided OracleVectorizerPreference. Additional
    index parameters can be supplied in params["parameters"]. Some fields are
    reserved and must not be present in params["parameters"]:
    - model, embedder_spec, vector_idxtype, vectorizer

    Optional clauses:
    - filter_by: list[str] -> FILTER BY clause
    - order_by: list[str] with order_by_asc -> ORDER BY ... ASC|DESC
    - parallel: int -> PARALLEL N

    Args:
        vectorizer_preference: Preference that defines the embedding configuration.
        idx_name: Name of the hybrid index (quoted as needed).
        params: dict of options including
            "parameters", "filter_by", "order_by", "order_by_asc", "parallel".

    Returns:
        str: DDL statement text.

    Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html

    """
    idx_name = _quote_identifier(idx_name)
    table_name = _quote_identifier(table_name)
    index_parameters = {}
    reserved_parameters = ["model", "embedder_spec", "vector_idxtype", "vectorizer"]
    for key, value in params.get("parameters", {}).items():
        _validate_identifier(key)
        normalized_key = key.strip()
        if normalized_key.lower() in reserved_parameters:
            raise ValueError(
                "Vectorization parameters must be given with OracleVectorizerPreference: "
                "do not include any of {model, embedder_spec, vector_idxtype, vectorizer} "
                "under params['parameters']."
            )
        index_parameters[normalized_key] = value

    params_str, filter_by_str, order_by_str, parallel_str = "", "", "", ""

    params_str = f"vectorizer {vectorizer_preference.preference_name} "
    for k, v in index_parameters.items():
        # Keys and values are trusted Oracle hybrid-vector parameter grammar
        # inside an escaped SQL string literal, not outer SQL syntax.
        params_str += f"{k} {v} "

    filter_by = params.get("filter_by")
    if filter_by:
        filter_by_str = (
            "FILTER BY " + _quote_identifier_list(filter_by, "filter_by") + " "
        )

    order_by = params.get("order_by")
    order_by_asc = params.get("order_by_asc", True)
    if not isinstance(order_by_asc, bool):
        raise ValueError("order_by_asc must be a boolean")
    if order_by:
        order_by_str = (
            "ORDER BY "
            + _quote_identifier_list(order_by, "order_by")
            + f" {'ASC' if order_by_asc else 'DESC'} "
        )

    parallel = params.get("parallel")
    if parallel is not None:
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel <= 0:
            raise ValueError("parallel must be a positive integer")
        parallel_str = f"PARALLEL {parallel} "

    def oracle_string_literal(value: str) -> str:
        return value.replace("'", "''")

    return f"""
    CREATE HYBRID VECTOR INDEX {idx_name} ON
    {table_name}(text)
    PARAMETERS ('{oracle_string_literal(params_str)}') {filter_by_str} {order_by_str} {parallel_str}
    """  # noqa E501


def _validate_identifier(name: str) -> None:
    _quote_identifier(name)


@_handle_exceptions
def create_hybrid_index(
    client: Union[Connection, ConnectionPool],
    idx_name: str,
    vector_store: OraLlamaVS,
    vectorizer_preference: Optional[OracleVectorizerPreference] = None,
    embeddings: Optional[OracleEmbeddings] = None,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """
    Create a HYBRID VECTOR INDEX if it does not already exist.

    The index uses either the provided OracleVectorizerPreference or, if
    `embeddings` is provided, a temporary preference derived from that
    OracleEmbeddings configuration (which is dropped after index creation).
    Additional options are accepted via params:
    - parameters: dict of INDEX PARAMETERS (excluding model/embedder_spec/vectorizer)
    - filter_by: list[str] -> FILTER BY clause
    - order_by: list[str], order_by_asc: bool -> ORDER BY ... ASC|DESC
    - parallel: int -> PARALLEL N

    Args:
        client: oracledb.Connection or oracledb.ConnectionPool.
        idx_name: Index name to create (quote if needed).
        vector_store: OraLlamaVS instance whose table will be indexed.
        vectorizer_preference: Existing OracleVectorizerPreference to reference
            in the index. Mutually exclusive with `embeddings`.
        embeddings: OracleEmbeddings to derive a temporary preference from.
            Mutually exclusive with `vectorizer_preference`.
        params: Optional dict of index options.

    Reference:
        https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html

    """
    if (not embeddings and not vectorizer_preference) or (
        embeddings and vectorizer_preference
    ):
        raise ValueError(
            "Exactly one of 'embeddings' or 'vectorizer_preference' must be provided."
        )

    drop = False
    if embeddings:
        vectorizer_preference = OracleVectorizerPreference.create_preference(
            client, embeddings
        )
        drop = True

    vectorizer_preference = cast(
        OracleVectorizerPreference,
        vectorizer_preference,
    )

    _validate_identifier(idx_name)
    _validate_identifier(vector_store.table_name)
    ddl = _get_hybrid_index_ddl(
        vectorizer_preference, idx_name, vector_store.table_name, params or {}
    )

    with _get_connection(client) as connection:
        if not _index_exists(connection, idx_name):
            with connection.cursor() as cur:
                cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")

    if drop:
        vectorizer_preference.drop_preference()
