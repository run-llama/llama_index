"""
Helper classes and constants for Azure CosmosDB NoSQL Vector Store.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class Constants:
    """Constants for Azure CosmosDB NoSQL Vector Store."""

    # Field names
    ID = "id"
    METADATA = "metadata"
    SIMILARITY_SCORE = "SimilarityScore"

    # Parameter Keys
    LIMIT = "limit"
    NAME = "name"
    VALUE = "value"
    VECTOR_KEY = "vectorKey"
    TEXT_KEY = "textKey"
    METADATA_KEY = "metadataKey"
    WEIGHTS = "weights"
    VECTOR_PARAM = "vector"
    FILTER_VALUE = "filter_value"

    # Full text rank filter
    SEARCH_FIELD = "search_field"
    SEARCH_TEXT = "search_text"


class AzureCosmosDBNoSqlVectorSearchType(str, Enum):
    """
    Vector store search type.

    Behavioural knobs are passed as separate keyword arguments rather than
    encoded into the search type:

    * ``threshold`` (float, optional) — when provided alongside ``VECTOR`` or
      ``HYBRID``, post-filter results by the per-row similarity / distance
      score (direction depends on the container's distance function).
    * ``weights`` (list of float, optional) — when provided alongside
      ``HYBRID``, runs server-side weighted RRF.
    """

    VECTOR = "vector"
    FULL_TEXT_SEARCH = "full_text_search"
    FULL_TEXT_RANKING = "full_text_ranking"
    HYBRID = "hybrid"


class ParamMapping:
    """Parameter mapping class for building parameterized queries."""

    class Parameter:
        """Internal parameter class for query parameters."""

        def __init__(self, key: str, value: Any) -> None:
            self.key = key
            self.value = value

    def __init__(
        self,
        table: str,
        name_key: str = Constants.NAME,
        value_key: str = Constants.VALUE,
    ) -> None:
        self.table = table
        self.name_key = name_key
        self.value_key = value_key
        self.parameter_map: Dict[str, ParamMapping.Parameter] = {}

    def add_parameter(self, key: str, value: Any) -> None:
        param_key = f"@{key}"
        self.parameter_map[key] = ParamMapping.Parameter(key=param_key, value=value)

    def gen_proj_field(self, key: str, value: Any, alias: Optional[str] = None) -> str:
        if key not in self.parameter_map:
            self.add_parameter(key, value)
        projection = f"{self.table}[{self.parameter_map[key].key}]"
        if alias:
            projection += f" as {alias}"
        return projection

    def gen_param_key(self, key: str, value: Any) -> str:
        if key not in self.parameter_map:
            self.add_parameter(key, value)
        return self.parameter_map[key].key

    def gen_vector_distance_proj_field(
        self,
        vector_field: str,
        vector: Any,
        alias: Optional[str] = None,
    ) -> str:
        vector_key = self.gen_param_key(key=Constants.VECTOR_KEY, value=vector_field)
        vector_param_key = self.gen_param_key(key=Constants.VECTOR_PARAM, value=vector)
        projection = f"VectorDistance({self.table}[{vector_key}], {vector_param_key})"
        if alias:
            projection += f" as {alias}"
        return projection

    def gen_vector_distance_order_by_field(
        self,
        vector_field: str,
        vector: Any,
        alias: Optional[str] = None,
    ) -> str:
        """
        Generate VectorDistance using direct field path and inline vector literal.

        Both the bracket indexer (table[@key]) AND query parameters (@vector) are
        rejected by CosmosDB inside ORDER BY RANK RRF. Use direct field path and
        inline the vector as an array literal instead.
        """
        vector_literal = "[" + ", ".join(str(v) for v in vector) + "]"
        projection = f"VectorDistance({self.table}.{vector_field}, {vector_literal})"
        if alias:
            projection += f" as {alias}"
        return projection

    def export_parameter_list(self) -> List[Dict[str, Any]]:
        return [
            {self.name_key: param.key, self.value_key: param.value}
            for param in self.parameter_map.values()
        ]
