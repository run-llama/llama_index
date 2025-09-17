"""SQL wrapper around SQLDatabase in langchain."""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import MetaData, create_engine, insert, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError


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

    def _extract_cte_names(self, command: str) -> set[str]:
        """
        Extract CTE names from a SQL command.

        Returns a set of CTE names that should not be schema-prefixed.
        """
        cte_names = set()

        # Pattern to match CTE definitions: WITH name AS (...), name2 AS (...)
        # This handles both single and multiple CTEs
        cte_pattern = r"\bWITH\s+(.+?)\s+AS\s*\("

        # Find the WITH clause
        with_match = re.search(cte_pattern, command, re.IGNORECASE | re.DOTALL)
        if not with_match:
            return cte_names

        # Extract everything between WITH and the first AS (
        with_clause_start = with_match.start()

        # Find all CTE definitions by looking for patterns like "name AS ("
        # This is more complex because we need to handle nested parentheses
        remaining_sql = command[with_clause_start:]

        # Simple approach: find all patterns like "identifier AS (" at the beginning or after commas
        cte_def_pattern = r"(?:WITH\s+|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\("

        for match in re.finditer(cte_def_pattern, remaining_sql, re.IGNORECASE):
            cte_name = match.group(1).strip()
            cte_names.add(cte_name)

        return cte_names

    def _add_schema_prefix_smart(self, command: str) -> str:
        """
        Add schema prefix to table names while preserving CTE names.

        This method:
        1. Identifies CTE names in the query
        2. Only adds schema prefixes to actual table names, not CTE names
        3. Handles both FROM and JOIN clauses properly
        """
        if not self._schema:
            return command

        # Extract CTE names that should not be prefixed
        cte_names = self._extract_cte_names(command)

        # Create a modified command
        modified_command = command

        # Pattern to find FROM/JOIN clauses followed by table names
        # This pattern captures: FROM/JOIN + whitespace + table_name
        from_join_pattern = (
            r"\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        )

        def replace_table_reference(match):
            clause_type = match.group(1)  # FROM or JOIN
            table_ref = match.group(2)  # table name or schema.table

            # Skip if this is a CTE name
            if table_ref in cte_names:
                return match.group(0)  # Return original match unchanged

            # Skip if already schema-qualified (contains a dot)
            if "." in table_ref:
                return match.group(0)  # Return original match unchanged

            # Add schema prefix
            return f"{clause_type} {self._schema}.{table_ref}"

        # Apply the replacement
        modified_command = re.sub(
            from_join_pattern,
            replace_table_reference,
            modified_command,
            flags=re.IGNORECASE,
        )

        return modified_command

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """
        Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            try:
                # Use smart schema prefixing that handles CTEs properly
                modified_command = self._add_schema_prefix_smart(command)
                cursor = connection.execute(text(modified_command))
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
