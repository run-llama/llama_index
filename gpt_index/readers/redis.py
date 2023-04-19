"""Redis Reader"""

from typing import Any, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

import numpy as np

def get_redis_query(
    return_fields: List[str],
    top_k: int = 20,
    vector_field_name: str = "vector",
    vector_param_name: str = "vector",
    sort: bool = True,
    filters: str = "*",
    ) -> List[Document]:
    """Create a vector query for use with a SearchIndex

    Args:
        return_fields (t.List[str]): A list of fields to return in the query results
        top_k (int, optional): The number of results to return. Defaults to 20.
        vector_field_name (str, optional): The name of the vector field in the index. Defaults to "vector".
        vector_param_name (str, optional): The name of the query param for searches. Defaults to "vector".
        sort (bool, optional): Whether to sort the results by score. Defaults to True.
        filters (str, optional): string to filter the results by. Defaults to "*".

    """
    from redis.commands.search.query import Query  # noqa: F401

    base_query = f"{filters}=>[KNN {top_k} @{vector_field_name} ${vector_param_name} AS vector_score]"
    query = Query(base_query).return_fields(*return_fields).dialect(2)
    if sort:
        query.sort_by("vector_score")
    return query


class RedisReader(BaseReader):
    """Redis Reader

    Args:
        host (str): Redis host.
        port (int): Redis port.
        password (str): Redis password.
        username (str): Redis username.
    """

    def __init__(self, host: str, port: int, password: str, username: str):
        """Initialize with parameters."""
        try:
            import redis # noqa: F401
            from redis.commands.search.query import Query  # noqa: F401
        except ImportError:
            raise ImportError(
                "`redis` package not found, please run `pip install redis`"
            )

        self._host = host
        self._port = port
        self._password = password
        self._username = username

        self._client = Redis(host=host, port=port, password=password, username=username)


    def load_data(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 20,
        return_values: bool = True,
        return_score: bool = True,
        filters: str = "*",
        ) -> List[Document]:
        """Create a vector query for use with a SearchIndex

        Args:
            return_fields (t.List[str]): A list of fields to return in the query results
            top_k (int, optional): The number of results to return. Defaults to 20.
            return_score (bool, optional): Whether to return the score in the query results. Defaults to True.
            sort (bool, optional): Whether to sort the results by score. Defaults to True.
            filters (str, optional): string to filter the results by. Defaults to "*".

        """
        return_fields = [
            "id",
            "text"
        ]
        if return_values:
            return_fields.append("vector")
        if return_score:
            return_fields.append("vector_score")

        query = redis_query(
            return_fields=return_fields,
            top_k=top_k,
            vector_field_name="vector",
            vector_param_name="vector",
            sort=True,
            filters=filters
        )
        query_params = {"vector": np.array(query_vector).astype(np.float32).tobytes()}
        docs = self._client.ft(index_name).search(query, query_params=query_params)

        results = []
        for doc in docs.docs:
            results.append(Document(text=doc.text, vector=np.frombuffer(doc.vector, dtype=np.float32).tolist()))

        return results
