"""Query schema."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs import IndexStructType


class QueryMode(str, Enum):
    """Query mode."""

    DEFAULT = "default"
    RETRIEVE = "retrieve"
    EMBEDDING = "embedding"

    # to hiearchically summarize using tree
    SUMMARIZE = "summarize"

    # for keyword extractor
    SIMPLE = "simple"
    RAKE = "rake"

    # recursive queries (composable queries)
    RECURSIVE = "recursive"


@dataclass
class QueryConfig(DataClassJsonMixin):
    """Query config.

    A list of query config objects is passed during a query call to define
    configurations for each individual subindex within an overall composed index.

    Args:
        index_struct_type (IndexStructType): The type of index struct.
        query_mode (QueryMode): The query mode.
        query_kwargs (Dict[str, Any], optional): The query kwargs. Defaults to {}.

    """

    index_struct_type: IndexStructType
    query_mode: QueryMode
    query_kwargs: Dict[str, Any] = field(default_factory=dict)
