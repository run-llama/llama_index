"""Snowflake Query Engine Pack."""

import os
from typing import Any, Dict, List

from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.core.llama_pack.base import BaseLlamaPack
from sqlalchemy import create_engine


class SnowflakeQueryEnginePack(BaseLlamaPack):
    """
    Snowflake query engine pack.
    It uses snowflake-sqlalchemy to connect to Snowflake, then calls
    NLSQLTableQueryEngine to query data.
    """

    def __init__(
        self,
        user: str,
        password: str,
        account: str,
        database: str,
        schema: str,
        warehouse: str,
        role: str,
        tables: List[str],
        **kwargs: Any,
    ) -> None:
        """Init params."""
        # workaround for https://github.com/snowflakedb/snowflake-sqlalchemy/issues/380.
        try:
            snowflake_sqlalchemy_20_monkey_patches()
        except Exception:
            raise ImportError("Please run `pip install snowflake-sqlalchemy`")

        if not os.environ.get("OPENAI_API_KEY", None):
            raise ValueError("OpenAI API Token is missing or blank.")

        snowflake_uri = f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}"

        engine = create_engine(snowflake_uri)

        self._sql_database = SQLDatabase(engine)
        self.tables = tables

        self.query_engine = NLSQLTableQueryEngine(
            sql_database=self._sql_database, tables=self.tables
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "sql_database": self._sql_database,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)


def snowflake_sqlalchemy_20_monkey_patches():
    import sqlalchemy.util.compat

    # make strings always return unicode strings
    sqlalchemy.util.compat.string_types = (str,)
    sqlalchemy.types.String.RETURNS_UNICODE = True

    import snowflake.sqlalchemy.snowdialect

    snowflake.sqlalchemy.snowdialect.SnowflakeDialect.returns_unicode_strings = True

    # make has_table() support the `info_cache` kwarg
    import snowflake.sqlalchemy.snowdialect

    def has_table(self, connection, table_name, schema=None, info_cache=None):
        """
        Checks if the table exists.
        """
        return self._has_object(connection, "TABLE", table_name, schema)

    snowflake.sqlalchemy.snowdialect.SnowflakeDialect.has_table = has_table
