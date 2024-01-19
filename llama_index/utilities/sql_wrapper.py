"""SQL wrapper around SQLDatabase in langchain."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import MetaData, create_engine, insert, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError


class SQLDatabase:
    """SQL Database.

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
        template = (
            "Table '{table_name}' has columns: {columns}, "
            "and foreign keys: {foreign_keys}."
        )
        columns = []
        for column in self._inspector.get_columns(table_name, schema=self._schema):
            if column.get("comment"):
                columns.append(
                    f"{column['name']} ({column['type']!s}): "
                    f"'{column.get('comment')}'"
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
        foreign_key_str = ", ".join(foreign_keys)
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

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            try:
                if self._schema:
                    command = command.replace("FROM ", f"FROM {self._schema}.")
                cursor = connection.execute(text(command))
            except (ProgrammingError, OperationalError) as exc:
                raise NotImplementedError(
                    f"Statement {command!r} is invalid SQL."
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
