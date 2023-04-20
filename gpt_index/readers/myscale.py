"""MyScale reader."""

import logging
from typing import Any, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

logger = logging.getLogger(__name__)

BS = "\\"
must_escape = (BS, "'")


class MyScaleSettings:
    """MyScale Client Configuration

    Attribute:
        table (str) : Table name to operate on.
        database (str) : Database name to find the table.
        index_type (str): index type string
        metric (str) : Metric to compute distance, supported are ('l2', 'cosine', 'ip'). Defaults to 'cosine'.
        index_params (Optional[dict]): index build parameter
    """

    def __init__(
        self,
        table: str,
        database: str,
        index_type: str,
        metric: str,
        index_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.table = table
        self.database = database
        self.index_type = index_type
        self.metric = metric
        self.index_params = index_params


def _escape_str(value: Optional[str]) -> str:
    return (
        "".join(f"{BS}{c}" if c in must_escape else c for c in value) if value else ""
    )


def _build_query_statement(
    config: MyScaleSettings,
    query_embed: List[float],
    where_str: Optional[str] = None,
    limit: Optional[int] = None,
    search_params: Optional[dict] = None,
) -> str:
    query_embed_str = ",".join(str(num) for num in query_embed)
    if where_str is not None:
        where_str = f"WHERE {where_str}"
    else:
        where_str = ""

    if config.metric.lower() == "ip":
        order = "DESC"
    else:
        order = "ASC"

    search_params_str = (
        "(" + ",".join([f"'{k}={v}'" for k, v in search_params.items()]) + ")"
        if search_params
        else ""
    )

    query_statement = f"""
        SELECT id, doc_id, text, node_info, extra_info, distance{search_params_str}(vector, [{query_embed_str}]) AS dist
        FROM {config.database}.{config.table} {where_str}
        ORDER BY dist {order}
        LIMIT {limit}
        """
    return query_statement


class MyscaleReader(BaseReader):
    """MyScale reader.

    Args:
        myscale_host (str) : An URL to connect to MyScale backend.
        username (str) : Usernamed to login.
        password (str) : Password to login.
        myscale_port (int) : URL port to connect with HTTP. Defaults to 8443.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on. Defaults to 'vector_table'.
        index_type (str): index type string. Default to "IVFLAT"
        index_param (dict): index build parameter. Default to None


    """

    def __init__(
        self,
        myscale_host: str,
        username: str,
        password: str,
        myscale_port: Optional[int] = 8443,
        database: str = "default",
        table: str = "llama_index",
        index_type: str = "IVFLAT",
        metric: str = "cosine",
        index_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = "`clickhouse_connect` package not found, please run `pip install clickhouse-connect`"
        try:
            import clickhouse_connect  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self.client = clickhouse_connect.get_client(
            host=myscale_host,
            port=myscale_port,
            username=username,
            password=password,
            **kwargs,
        )

        self.config = MyScaleSettings(
            table=table,
            database=database,
            index_type=index_type,
            metric=metric,
            index_params=index_params,
        )

    def load_data(
        self,
        query_vector: List[float],
        config: MyScaleSettings,
        where_str: Optional[str] = None,
        search_params: Optional[dict] = None,
        limit: int = 10,
    ) -> List[Document]:
        """Load data from MyScale.

        Args:
            query_vector (List[float]): Query vector.
            table_name (str): Name of the MyScale table.
            where_str (Optional[str], optional): where condition string. Defaults to None.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.
        """

        query_statement = _build_query_statement(
            config=config,
            query_embed=query_vector,
            where_str=where_str,
            limit=limit,
            search_params=search_params,
        )

        try:
            return [
                Document(doc_id=r["id"], text=r["text"])
                for r in self.client.query(query_statement).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            raise e
