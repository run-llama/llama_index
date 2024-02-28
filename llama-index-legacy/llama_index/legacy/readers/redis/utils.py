import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern

import numpy as np

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.query import Query


class TokenEscaper:
    """
    Escape punctuation within an input string. Taken from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match: re.Match) -> str:
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20400},
    {"name": "searchlight", "ver": 20400},
]


def check_redis_modules_exist(client: "RedisType") -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in REDIS_REQUIRED_MODULES:
        if module["name"] in installed_modules and int(
            installed_modules[module["name"]][b"ver"]
        ) >= int(
            module["ver"]
        ):  # type: ignore[call-overload]
            return
    # otherwise raise error
    error_message = (
        "You must add the RediSearch (>= 2.4) module from Redis Stack. "
        "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
    )
    _logger.error(error_message)
    raise ValueError(error_message)


def get_redis_query(
    return_fields: List[str],
    top_k: int = 20,
    vector_field: str = "vector",
    sort: bool = True,
    filters: str = "*",
) -> "Query":
    """Create a vector query for use with a SearchIndex.

    Args:
        return_fields (t.List[str]): A list of fields to return in the query results
        top_k (int, optional): The number of results to return. Defaults to 20.
        vector_field (str, optional): The name of the vector field in the index.
            Defaults to "vector".
        sort (bool, optional): Whether to sort the results by score. Defaults to True.
        filters (str, optional): string to filter the results by. Defaults to "*".

    """
    from redis.commands.search.query import Query

    base_query = f"{filters}=>[KNN {top_k} @{vector_field} $vector AS vector_score]"

    query = Query(base_query).return_fields(*return_fields).dialect(2).paging(0, top_k)

    if sort:
        query.sort_by("vector_score")
    return query


def convert_bytes(data: Any) -> Any:
    if isinstance(data, bytes):
        return data.decode("ascii")
    if isinstance(data, dict):
        return dict(map(convert_bytes, data.items()))
    if isinstance(data, list):
        return list(map(convert_bytes, data))
    if isinstance(data, tuple):
        return map(convert_bytes, data)
    return data


def array_to_buffer(array: List[float], dtype: Any = np.float32) -> bytes:
    return np.array(array).astype(dtype).tobytes()
