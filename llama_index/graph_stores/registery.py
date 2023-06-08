from enum import Enum
from typing import Dict, Type

from llama_index.graph_stores.nebulagraph import NebulaGraphStore
from llama_index.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.types import GraphStore


class GraphStoreType(str, Enum):
    SIMPLE = "simple_kg"
    NEBULA = "nebulagraph"


GRAPH_STORE_TYPE_TO_GRAPH_STORE_CLASS: Dict[GraphStoreType, Type[GraphStore]] = {
    GraphStoreType.SIMPLE: SimpleGraphStore,
    GraphStoreType.NEBULA: NebulaGraphStore,
}

GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE: Dict[Type[GraphStore], GraphStoreType] = {
    cls_: type_ for type_, cls_ in GRAPH_STORE_TYPE_TO_GRAPH_STORE_CLASS.items()
}
