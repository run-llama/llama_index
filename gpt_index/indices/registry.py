"""Index registry."""

from dataclasses import dataclass, field
from typing import Dict, Type

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.data_structs import (
    IndexDict,
    IndexGraph,
    IndexList,
    IndexStruct,
    KeywordTable,
    PineconeIndexStruct,
    SimpleIndexDict,
    WeaviateIndexStruct,
)
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.data_structs.table import SQLStructTable
from gpt_index.indices.query.base import BaseGPTIndexQuery

# DEFAULT_TYPE_TO_STRUCT = {
#     IndexStructType.TREE: IndexGraph,
#     IndexStructType.LIST: IndexList,
#     IndexStructType.KEYWORD_TABLE: KeywordTable,
#     IndexStructType.DICT: IndexDict,
#     IndexStructType.SIMPLE_DICT: SimpleIndexDict,
#     IndexStructType.WEAVIATE: WeaviateIndexStruct,
#     IndexStructType.PINECONE: PineconeIndexStruct,
#     IndexStructType.SQL: SQLStructTable,
# }


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

    # def __init__(self) -> None:
    #     """Init params."""
    #     self.type_to_struct: Dict[str, Type[IndexStruct]] = field(default_factory=dict)
    #     self.type_to_query: Dict[str, Type[BaseGPTIndexQuery]] = field(
    #         default_factory=dict
    #     )
