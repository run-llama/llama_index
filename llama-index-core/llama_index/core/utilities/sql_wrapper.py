"""SQL wrapper around SQLDatabase in langchain."""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy import MetaData, Table, create_engine, inspect, insert, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError,SQLAlchemyError

import re
import uuid
from collections import defaultdict




class SQLDatabase:
    """
    SQL Database.

    This class provides a wrapper around the SQLAlchemy engine to interact with a SQL
    database.
    It provides methods to execute SQL commands, insert data into tables, and retrieve
    information about the database schema.
    It also supports optional features such as including or excluding specific tables,
    sampling rows for table info,
    including indexes in table info, and supporting views.

    Based on langchain SQLDatabase.
    https://github.com/langchain-ai/langchain/blob/e355606b1100097665207ca259de6dc548d44c78/libs/langchain/langchain/utilities/sql_database.py#L39

    Args:
        engine (Engine): The SQLAlchemy engine instance to use for database operations.
        schema (Optional[str]): The name of the schema to use, if any.
        metadata (Optional[MetaData]): The metadata instance to use, if any.
        ignore_tables (Optional[List[str]]): List of table names to ignore. If set,
            include_tables must be None.
        include_tables (Optional[List[str]]): List of table names to include. If set,
            ignore_tables must be None.
        sample_rows_in_table_info (int): The number of sample rows to include in table
            info.
        indexes_in_table_info (bool): Whether to include indexes in table info.
        custom_table_info (Optional[dict]): Custom table info to use.
        view_support (bool): Whether to support views.
        max_string_length (int): The maximum string length to use.

    """

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 300,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)

        # including view support by adding the views as well as tables to the all
        # tables list if view_support is True
        self._all_tables = set(
            self._inspector.get_table_names(schema=schema)
            + (self._inspector.get_view_names(schema=schema) if view_support else [])
        )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = {
                table: info
                for table, info in self._custom_table_info.items()
                if table in intersection
            }

        self._max_string_length = max_string_length

        self._metadata = metadata or MetaData()
        # including view support if view_support = true
        self._metadata.reflect(
            views=view_support,
            bind=self._engine,
            only=list(self._usable_tables),
            schema=self._schema,
        )

    @property
    def engine(self) -> Engine:
        """Return SQL Alchemy engine."""
        return self._engine

    @property
    def metadata_obj(self) -> MetaData:
        """Return SQL Alchemy metadata."""
        return self._metadata

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> "SQLDatabase":
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    def get_table_columns(self, table_name: str) -> List[Any]:
        """Get table columns."""
        return self._inspector.get_columns(table_name)

    def get_single_table_info(self, table_name: str) -> str:
        """Get table info for a single table."""
        # same logic as table_info, but with specific table names
        template = "Table '{table_name}' has columns: {columns}, "
        try:
            # try to retrieve table comment
            table_comment = self._inspector.get_table_comment(
                table_name, schema=self._schema
            )["text"]
            if table_comment:
                template += f"with comment: ({table_comment}) "
        except NotImplementedError:
            # get_table_comment raises NotImplementedError for a dialect that does not support comments.
            pass

        template += "{foreign_keys}."
        columns = []
        for column in self._inspector.get_columns(table_name, schema=self._schema):
            if column.get("comment"):
                columns.append(
                    f"{column['name']} ({column['type']!s}): '{column.get('comment')}'"
                )
            else:
                columns.append(f"{column['name']} ({column['type']!s})")

        column_str = ", ".join(columns)
        foreign_keys = []
        for foreign_key in self._inspector.get_foreign_keys(
            table_name, schema=self._schema
        ):
            foreign_keys.append(
                f"{foreign_key['constrained_columns']} -> "
                f"{foreign_key['referred_table']}.{foreign_key['referred_columns']}"
            )
        foreign_key_str = (
            foreign_keys
            and " and foreign keys: {}".format(", ".join(foreign_keys))
            or ""
        )
        return template.format(
            table_name=table_name, columns=column_str, foreign_keys=foreign_key_str
        )

    def insert_into_table(self, table_name: str, data: dict) -> None:
        """Insert data into a table."""
        table = self._metadata.tables[table_name]
        stmt = insert(table).values(**data)
        with self._engine.begin() as connection:
            connection.execute(stmt)

    def truncate_word(self, content: Any, *, length: int, suffix: str = "...") -> str:
        """
        Truncate a string to a certain number of words, based on the max string
        length.
        """
        if not isinstance(content, str) or length <= 0:
            return content

        if len(content) <= length:
            return content

        return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix

    def _add_schema_prefix(self, command: str) -> str:
        """
        Add schema prefix to table names in FROM/JOIN clauses.

        Preserves CTE (Common Table Expression) names and already
        schema-qualified identifiers so they are not double-prefixed.
        """
        # Collect CTE names defined in WITH clauses
        cte_names: Set[str] = set()
        # First CTE: WITH [RECURSIVE] name AS (
        for m in re.finditer(
            r"\bWITH\s+(?:RECURSIVE\s+)?(\w+)\s+AS\s*\(", command, re.IGNORECASE
        ):
            cte_names.add(m.group(1).lower())
        # Subsequent CTEs: ), name AS (
        for m in re.finditer(r"\)\s*,\s*(\w+)\s+AS\s*\(", command, re.IGNORECASE):
            cte_names.add(m.group(1).lower())

        def _replace(match: re.Match) -> str:
            keyword = match.group(1)
            table_ref = match.group(2)
            # Skip CTE references and already schema-qualified names
            if table_ref.lower() in cte_names or "." in table_ref:
                return match.group(0)
            return f"{keyword}{self._schema}.{table_ref}"

        return re.sub(
            r"\b((?:FROM|JOIN)\s+)(\w+(?:\.\w+)?)",
            _replace,
            command,
            flags=re.IGNORECASE,
        )

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """
        Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            try:
                if self._schema:
                    command = self._add_schema_prefix(command)
                cursor = connection.execute(text(command))
            except (ProgrammingError, OperationalError) as exc:
                raise NotImplementedError(
                    f"Statement {command!r} is invalid SQL.\nError: {exc.orig}"
                ) from exc
            if cursor.returns_rows:
                result = cursor.fetchall()
                # truncate the results to the max string length
                # we can't use str(result) directly because it automatically truncates long strings
                truncated_results = []
                for row in result:
                    # truncate each column, then convert the row to a tuple
                    truncated_row = tuple(
                        self.truncate_word(column, length=self._max_string_length)
                        for column in row
                    )
                    truncated_results.append(truncated_row)
                return str(truncated_results), {
                    "result": truncated_results,
                    "col_keys": list(cursor.keys()),
                }
        return "", {}


class TrinoSQLDatabase(SQLDatabase):
    """
    SQLDatabase subclass for Trino multi-catalog environments.

    Bypasses all SQLAlchemy inspector calls (which are single-catalog bound)
    and builds MetaData manually by querying information_schema across catalogs.

    All query/insert/truncate methods are inherited from SQLDatabase unchanged.
    Only schema-introspection methods are overridden.
    """

    def __init__(
        self,
        engine: Engine,
        catalogs: List[str],
        default_schema: str = "public",
        max_string_length: int = 300,
    ):
        # Deliberately skip SQLDatabase.__init__ entirely —
        # it calls inspector + reflect which both fail on multi-catalog Trino.
        self._engine = engine
        self._schema = default_schema
        self._catalogs = catalogs
        self._default_schema = default_schema
        self._max_string_length = max_string_length
        self._sample_rows_in_table_info = 0
        self._indexes_in_table_info = False
        self._custom_table_info = None

        # Build our cross-catalog metadata without any inspector
        self._metadata = self._build_global_metadata()

        # _usable_tables and _all_tables derived from what we actually reflected
        self._all_tables: Set[str] = set(self._metadata.tables.keys())
        self._usable_tables: Set[str] = self._all_tables
        self._include_tables: Set[str] = set()
        self._ignore_tables: Set[str] = set()

    # ------------------------------------------------------------------
    # Internal metadata builder (your original logic, encapsulated)
    # ------------------------------------------------------------------

    def _fetch_all_catalog_tables(self) -> List[Dict[str, Any]]:
        """Query information_schema across all catalogs in one UNION query."""
        union_parts = [
            f"SELECT '{cat}' AS catalog, table_schema, table_name "
            f"FROM {cat}.information_schema.tables "
            f"WHERE table_schema = '{self._default_schema}' "
            f"AND table_type IN ('BASE TABLE', 'VIEW')"
            for cat in self._catalogs
        ]
        info_query = (
            "SELECT catalog, table_schema, table_name, "
            "concat(catalog, '.', table_schema, '.', table_name) AS fq_name "
            "FROM (\n" + "\nUNION ALL\n".join(union_parts) + "\n) t\n"
            "ORDER BY catalog, table_name"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(text(info_query)).mappings().all()
            return [dict(r) for r in rows]

    def _build_trino_engine_for(self, catalog: str) -> Engine:
        """Build a single-catalog Trino engine by swapping the catalog in the URL."""
        url = self._engine.url
        # Trino URL format: trino://user:pass@host:port/catalog/schema
        # We rebuild it with the target catalog
        new_url = url.set(database=f"{catalog}/{self._default_schema}")
        return create_engine(new_url, pool_pre_ping=True)

    def _build_global_metadata(self) -> MetaData:
        """
        Build a single MetaData containing all tables across all catalogs.
        Keys are "catalog.schema.tablename".
        """
        rows = self._fetch_all_catalog_tables()

        by_catalog: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            by_catalog[r["catalog"]].append(r["table_name"])

        global_meta = MetaData()

        for catalog, table_names in by_catalog.items():
            if not table_names:
                continue

            cat_engine = self._build_trino_engine_for(catalog)
            local_meta = MetaData()

            try:
                local_meta.reflect(
                    bind=cat_engine,
                    schema=self._default_schema,
                    only=table_names,
                )
            except SQLAlchemyError as e:
                print(f"[TrinoSQLDatabase] reflect failed for catalog={catalog}: {e}. Falling back to per-table autoload.")
                for name in table_names:
                    try:
                        Table(name, local_meta, schema=self._default_schema, autoload_with=cat_engine)
                    except SQLAlchemyError as e2:
                        print(f"[TrinoSQLDatabase]   Failed to autoload {catalog}.{self._default_schema}.{name}: {e2}")

            for tbl in list(local_meta.tables.values()):
                try:
                    tbl.to_metadata(global_meta, schema=f"{catalog}.{self._default_schema}")
                except Exception as e:
                    print(f"[TrinoSQLDatabase]   Failed to copy {catalog}.{self._default_schema}.{tbl.name}: {e}")

            cat_engine.dispose()

        return global_meta

    # ------------------------------------------------------------------
    # Overrides: replace all inspector-based methods with metadata-based ones
    # ------------------------------------------------------------------

    def get_usable_table_names(self) -> Iterable[str]:
        """Return all table keys from global metadata."""
        return sorted(self._all_tables)

    def get_table_columns(self, table_name: str) -> List[Any]:
        """Get columns from MetaData instead of inspector."""
        table = self._metadata.tables.get(table_name)
        if table is None:
            return []
        return [
            {
                "name": col.name,
                "type": col.type,
                "comment": col.comment,
                "nullable": col.nullable,
                "default": col.default,
            }
            for col in table.columns
        ]

    def get_single_table_info(self, table_name: str) -> str:
        """Build table info string from MetaData — no inspector, no engine calls."""
        table = self._metadata.tables.get(table_name)
        if table is None:
            return f"Table '{table_name}' not found in metadata."

        template = "Table '{table_name}' has columns: {columns}, "

        if table.comment:
            template += f"with comment: ({table.comment}) "

        template += "{foreign_keys}."

        columns = []
        for col in table.columns:
            if col.comment:
                columns.append(f"{col.name} ({col.type!s}): '{col.comment}'")
            else:
                columns.append(f"{col.name} ({col.type!s})")
        column_str = ", ".join(columns)

        foreign_keys = []
        for fk in table.foreign_keys:
            foreign_keys.append(
                f"['{fk.parent.name}'] -> "
                f"{fk.column.table.name}.['{fk.column.name}']"
            )
        foreign_key_str = (
            " and foreign keys: {}".format(", ".join(foreign_keys))
            if foreign_keys
            else ""
        )

        return template.format(
            table_name=table_name,
            columns=column_str,
            foreign_keys=foreign_key_str,
        )

    # ------------------------------------------------------------------
    # Properties (engine and metadata_obj inherited shape, just re-declared)
    # ------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def metadata_obj(self) -> MetaData:
        return self._metadata

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    # ------------------------------------------------------------------
    # All other methods (run_sql, insert_into_table, truncate_word,
    # _add_schema_prefix, from_uri) are inherited from SQLDatabase as-is.
    # ------------------------------------------------------------------