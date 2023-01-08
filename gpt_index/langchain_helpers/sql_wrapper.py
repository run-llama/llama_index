"""SQL wrapper around SQLDatabase in langchain."""
from typing import Any, Dict, List, Tuple

from langchain.sql_database import SQLDatabase as LangchainSQLDatabase
from sqlalchemy import MetaData, insert
from sqlalchemy.engine import Engine


class SQLDatabase(LangchainSQLDatabase):
    """SQL Database."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        # self.metadata_obj = MetaData(bind=self._engine, reflect=True)
        self.metadata_obj = MetaData(bind=self._engine)
        self.metadata_obj.reflect()

    @property
    def engine(self) -> Engine:
        """Return SQL Alchemy engine."""
        return self._engine

    def get_table_columns(self, table_name: str) -> List[dict]:
        """Get table columns."""
        return self._inspector.get_columns(table_name)

    def get_single_table_info(self, table_name: str) -> str:
        """Get table info for a single table."""
        template = "Table '{table_name}' has columns: {columns}."
        columns = []
        for column in self._inspector.get_columns(table_name):
            columns.append(f"{column['name']} ({str(column['type'])})")
        column_str = ", ".join(columns)
        table_str = template.format(table_name=table_name, columns=column_str)
        return table_str

    def insert_into_table(self, table_name: str, data: dict) -> None:
        """Insert data into a table."""
        table = self.metadata_obj.tables[table_name]
        stmt = insert(table).values(**data)
        self._engine.execute(stmt)

    def run_sql(self, command: str) -> Tuple[str, Dict]:
        """Execute a SQL statement and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.connect() as connection:
            cursor = connection.exec_driver_sql(command)
            if cursor.returns_rows:
                result = cursor.fetchall()
                return str(result), {"result": result}
        return "", {}
