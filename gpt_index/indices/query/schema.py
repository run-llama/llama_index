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

    # for keyword extractor
    SIMPLE = "simple"
    RAKE = "rake"


@dataclass
class QueryConfig(DataClassJsonMixin):
    """Query config."""

    index_struct_type: IndexStructType
    query_mode: QueryMode
    query_kwargs: Dict[str, Any] = field(default_factory=dict)
