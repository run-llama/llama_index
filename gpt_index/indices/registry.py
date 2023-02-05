"""Index registry."""

from dataclasses import dataclass, field
from typing import Dict, Type

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.indices.query.base import BaseGPTIndexQuery

# map from mode to query class
QUERY_MAP_TYPE = Dict[str, Type[BaseGPTIndexQuery]]


@dataclass
class IndexRegistry:
    """Index registry.

    Stores mapping from index type to index_struct + queries.
    NOTE: this cannot be easily serialized, so must be re-initialized
    each time.
    If the user defines custom IndexStruct or query classes,
    they must be added to the registry manually.

    """

    type_to_struct: Dict[str, Type[IndexStruct]] = field(default_factory=dict)
    type_to_query: Dict[str, QUERY_MAP_TYPE] = field(default_factory=dict)

    def update(self, other: "IndexRegistry") -> None:
        """Update the registry with another registry."""
        self.type_to_struct.update(other.type_to_struct)
        self.type_to_query.update(other.type_to_query)
