"""Tools for interacting with an Apache Cassandra database."""

from typing import List

from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.cassandra.cassandra_database_wrapper import (
    CassandraDatabase,
)


class CassandraDatabaseToolSpec(BaseToolSpec):
    """Base tool for interacting with an Apache Cassandra database."""

    db: CassandraDatabase = Field(exclude=True)

    spec_functions = [
        "cassandra_db_query",
        "cassandra_db_schema",
        "cassandra_db_select_table_data",
    ]

    def __init__(self, db: CassandraDatabase) -> None:
        """DB session in context."""
        self.db = db

    def cassandra_db_query(self, query: str) -> List[Document]:
        """
        Execute a CQL query and return the results as a list of Documents.

        Args:
            query (str): A CQL query to execute.

        Returns:
            List[Document]: A list of Document objects, each containing data from a row.

        """
        documents = []
        result = self.db.run_no_throw(query, fetch="Cursor")
        for row in result:
            doc_str = ", ".join([str(value) for value in row])
            documents.append(Document(text=doc_str))
        return documents

    def cassandra_db_schema(self, keyspace: str) -> List[Document]:
        """
        Input to this tool is a keyspace name, output is a table description
            of Apache Cassandra tables.
            If the query is not correct, an error message will be returned.
            If an error is returned, report back to the user that the keyspace
            doesn't exist and stop.

        Args:
            keyspace (str): The name of the keyspace for which to return the schema.

        Returns:
            List[Document]: A list of Document objects, each containing a table description.

        """
        return [Document(text=self.db.get_keyspace_tables_str(keyspace))]

    def cassandra_db_select_table_data(
        self, keyspace: str, table: str, predicate: str, limit: int
    ) -> List[Document]:
        """
        Tool for getting data from a table in an Apache Cassandra database.
            Use the WHERE clause to specify the predicate for the query that uses the
            primary key. A blank predicate will return all rows. Avoid this if possible.
            Use the limit to specify the number of rows to return. A blank limit will
            return all rows.

        Args:
            keyspace (str): The name of the keyspace containing the table.
            table (str): The name of the table for which to return data.
            predicate (str): The predicate for the query that uses the primary key.
            limit (int): The maximum number of rows to return.

        Returns:
            List[Document]: A list of Document objects, each containing a row of data.

        """
        return [
            Document(text=self.db.get_table_data(keyspace, table, predicate, limit))
        ]
