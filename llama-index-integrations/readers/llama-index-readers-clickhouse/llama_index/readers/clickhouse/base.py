"""ClickHouse reader."""
import logging
from typing import Any, List, Optional
import clickhouse_connect
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def escape_str(value: str) -> str:
    BS = "\\"
    must_escape = (BS, "'")
    return (
        "".join(f"{BS}{c}" if c in must_escape else c for c in value) if value else ""
    )


def format_list_to_string(lst: List) -> str:
    return "[" + ",".join(str(item) for item in lst) + "]"


DISTANCE_MAPPING = {
    "l2": "L2Distance",
    "cosine": "cosineDistance",
    "dot": "dotProduct",
}


class ClickHouseSettings:
    """ClickHouse Client Configuration.

    Attributes:
        table (str): Table name to operate on.
        database (str): Database name to find the table.
        engine (str): Engine. Options are "MergeTree" and "Memory". Default is "MergeTree".
        index_type (str): Index type string.
        metric (str): Metric type to compute distance e.g., cosine, l3, or dot.
        batch_size (int): The size of documents to insert.
        index_params (dict, optional): Index build parameter.
        search_params (dict, optional): Index search parameters for ClickHouse query.
    """

    def __init__(
        self,
        table: str,
        database: str,
        engine: str,
        index_type: str,
        metric: str,
        batch_size: int,
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.table = table
        self.database = database
        self.engine = engine
        self.index_type = index_type
        self.metric = metric
        self.batch_size = batch_size
        self.index_params = index_params
        self.search_params = search_params

    def build_query_statement(
        self,
        query_embed: List[float],
        where_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        query_embed_str = format_list_to_string(query_embed)
        where_str = f"WHERE {where_str}" if where_str else ""
        distance = DISTANCE_MAPPING[self.metric]
        return f"""
            SELECT id, doc_id, text, node_info, metadata,
            {distance}(vector, {query_embed_str}) AS score
            FROM {self.database}.{self.table} {where_str}
            ORDER BY score ASC
            LIMIT {limit}
            """


class ClickHouseReader(BaseReader):
    """ClickHouse reader.

    Args:
        clickhouse_host (str) : An URL to connect to ClickHouse backend. Default to "localhost".
        username (str) : Username to login. Defaults to "default".
        password (str) : Password to login. Defaults to "".
        clickhouse_port (int) : URL port to connect with HTTP. Defaults to 8123.
        database (str) : Database name to find the table. Defaults to 'default'.
        engine (str) : Engine. Options are "MergeTree" and "Memory". Default is "MergeTree".
        table (str) : Table name to operate on. Defaults to 'vector_table'.
        index_type (str): index type string. Default to "NONE", supported are ("NONE", "HNSW", "ANNOY")
        metric (str) : Metric to compute distance, supported are ('l2', 'cosine', 'dot').
            Defaults to 'cosine'
        batch_size (int, optional): the size of documents to insert. Defaults to 1000.
        index_params (dict, optional): The index parameters for ClickHouse.
            Defaults to None.
        search_params (dict, optional): The search parameters for a ClicKHouse query.
            Defaults to None.
    """

    def __init__(
        self,
        clickhouse_host: str = "localhost",
        username: str = "default",
        password: str = "",
        clickhouse_port: Optional[int] = 8123,
        database: str = "default",
        engine: str = "MergeTree",
        table: str = "llama_index",
        index_type: str = "NONE",
        metric: str = "cosine",
        batch_size: int = 1000,
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.client = clickhouse_connect.get_client(
            host=clickhouse_host,
            port=clickhouse_port,
            username=username,
            password=password,
        )

        self.config = ClickHouseSettings(
            table=table,
            database=database,
            engine=engine,
            index_type=index_type,
            metric=metric,
            batch_size=batch_size,
            index_params=index_params,
            search_params=search_params,
            **kwargs,
        )

    def load_data(
        self,
        query_vector: List[float],
        where_str: Optional[str] = None,
        limit: int = 10,
    ) -> List[Document]:
        """Load data from ClickHouse.

        Args:
            query_vector (List[float]): Query vector.
            where_str (Optional[str], optional): where condition string.
                Defaults to None.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.
        """
        query_statement = self.config.build_query_statement(
            query_embed=query_vector,
            where_str=where_str,
            limit=limit,
        )

        return [
            Document(id_=r["doc_id"], text=r["text"], metadata=r["metadata"])
            for r in self.client.query(query_statement).named_results()
        ]
