"""Query schema."""

from enum import Enum
from gpt_index.indices.data_structs import Node, IndexStruct, IndexStructType
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import List, Dict, Any, Optional


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

