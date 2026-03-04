"""Snowflake Reader."""

import logging
from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class SnowflakeReader(BaseReader):
    """
    Initializes a new instance of the SnowflakeReader.

    This class establishes a connection to Snowflake using SQLAlchemy, executes query
    and concatenates each row into Document used by LlamaIndex.

    Attributes:
        engine (Optional[Engine]): SQLAlchemy Engine object of the database connection.

        OR

        account (Optional[str]): Snowflake account identifier.
        user (Optional[str]): Snowflake account username.
        password (Optional[str]): Password for the Snowflake account.
        database (Optional[str]): Snowflake database name.
        schema (Optional[str]): Snowflake schema name.
        warehouse (Optional[str]): Snowflake warehouse name.
        proxy (Optional[str]): Proxy setting for the connection.

    """

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        proxy: Optional[str] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        """
        Initializes the SnowflakeReader with optional connection details, proxy configuration, or an engine directly.

        Args:
            account (Optional[str]): Snowflake account identifier.
            user (Optional[str]): Snowflake account username.
            password (Optional[str]): Password for the Snowflake account.
            database (Optional[str]): Snowflake database name.
            schema (Optional[str]): Snowflake schema name.
            warehouse (Optional[str]): Snowflake warehouse name.
            role (Optional[str]): Snowflake role name.
            proxy (Optional[str]): Proxy setting for the connection.
            engine (Optional[Engine]): Existing SQLAlchemy engine.

        """
        from snowflake.sqlalchemy import URL

        if engine is None:
            connect_args = {}
            if proxy:
                connect_args["proxy"] = proxy

            # Create an SQLAlchemy engine for Snowflake
            self.engine = create_engine(
                URL(
                    account=account or "",
                    user=user or "",
                    password=password or "",
                    database=database or "",
                    schema=schema or "",
                    warehouse=warehouse or "",
                    role=role or "",
                ),
                connect_args=connect_args,
            )
        else:
            self.engine = engine

        # Create a sessionmaker bound to the engine
        self.Session = sessionmaker(bind=self.engine)

    def execute_query(self, query_string: str) -> List[Any]:
        """
        Executes a SQL query and returns the fetched results.

        Args:
            query_string (str): The SQL query to be executed.

        Returns:
            List[Any]: The fetched results from the query.

        """
        # Create a session and execute the query
        session = self.Session()
        try:
            result = session.execute(text(query_string))
            return result.fetchall()
        finally:
            # Ensure the session is closed after query execution
            session.close()

    def load_data(self, query: str) -> List[Document]:
        """
        Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.

        """
        documents = []

        if query is None:
            raise ValueError("A query parameter is necessary to filter the data")

        try:
            result = self.execute_query(query)

            for item in result:
                # fetch each item
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(Document(text=doc_str))
            return documents
        except Exception as e:
            logger.error(
                f"An error occurred while loading the data: {e}", exc_info=True
            )
