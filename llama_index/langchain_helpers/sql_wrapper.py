"""SQL wrapper around SQLDatabase in langchain."""
from typing import Any, Dict, List, Tuple, Optional

from llama_index.bridge.langchain import SQLDatabase as LangchainSQLDatabase
from sqlalchemy import MetaData, create_engine, insert, text
from sqlalchemy.engine import Engine


class SQLDatabase(LangchainSQLDatabase):
    """SQL Database.

    Wrapper around SQLDatabase object from langchain. Offers
    some helper utilities for insertion and querying.
    See `langchain documentation <https://tinyurl.com/4we5ku8j>`_ for more details:

    Args:
        *args: Arguments to pass to langchain SQLDatabase.
        **kwargs: Keyword arguments to pass to langchain SQLDatabase.

    """

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
        for column in self._inspector.get_columns(table_name):
            if column.get("comment"):
                columns.append(
                    f"{column['name']} ({str(column['type'])}): '{column.get('comment')}'"
                )
            else:
                columns.append(f"{column['name']} ({str(column['type'])})")

        column_str = ", ".join(columns)
        foreign_keys = []
        for foreign_key in self._inspector.get_foreign_keys(table_name):
            foreign_keys.append(
                f"{foreign_key['constrained_columns']} -> "
                f"{foreign_key['referred_table']}.{foreign_key['referred_columns']}"
            )
        foreign_key_str = ", ".join(foreign_keys)
        table_str = template.format(
            table_name=table_name, columns=column_str, foreign_keys=foreign_key_str
        )
        return table_str

    def insert_into_table(self, table_name: str, data: dict) -> None:
        """Insert data into a table."""
        table = self._metadata.tables[table_name]
        stmt = insert(table).values(**data)
        with self._engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.connect() as connection:
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                result = cursor.fetchall()
                return str(result), {"result": result}
        return "", {}
