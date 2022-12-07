"""Query schema."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.indices.data_structs import IndexStruct, IndexStructType, Node


class QueryMode(Enum):
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
    query_kwargs: Dict[str, Any]
