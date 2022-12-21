"""Weaviate-specific serializers for GPT Index data structures.

Contain conversion to and from dataclasses that GPT Index uses.

"""

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, cast

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs import IndexGraph, IndexStruct, Node
from gpt_index.readers.weaviate.utils import get_by_id, validate_client
from gpt_index.utils import get_new_id

IS = TypeVar("IS", bound=IndexStruct)

DEFAULT_CLASS_PREFIX = "Gpt_Index"


class BaseWeaviateIndexStruct(Generic[IS]):
    """Base Weaviate index struct."""

    @classmethod
    @abstractmethod
    def _class_name(cls, class_prefix: str) -> str:
        """Return class name."""

    @classmethod
    def _get_common_properties(cls) -> List[Dict]:
        """Get common properties."""
        return [
            {
                "dataType": ["string"],
                "description": f"Text property",
                "name": "text",
            },
            {
                "dataType": ["string"],
                "description": f"Document id",
                "name": "doc_id",
            },
        ]

    @classmethod
    @abstractmethod
    def _get_properties(cls) -> List[Dict]:
        """Get properties specific to each index struct.

        Used in creating schema.

        """

    @classmethod
    def _get_by_id(cls, client: Any, object_id: str, class_prefix: str) -> Dict:
        """Get entry by id."""
        validate_client(client)
        class_name = cls._class_name(class_prefix)
        properties = cls._get_common_properties() + cls._get_properties()
        prop_names = [p["name"] for p in properties]
        entry = get_by_id(client, object_id, class_name, prop_names)
        return entry

    @classmethod
    def create_schema(cls, client: Any, class_prefix: str) -> None:
        """Create schema."""
        validate_client(client)
        # first check if schema exists
        schema = client.schema.get()
        classes = schema["classes"]
        existing_class_names = {c["class"] for c in classes}
        # if schema already exists, don't create
        class_name = cls._class_name(class_prefix)
        if class_name in existing_class_names:
            return

        # get common properties
        properties = cls._get_common_properties()
        # get specific properties
        properties.extend(cls._get_properties())
        class_obj = {
            "class": cls._class_name(class_prefix),  # <= note the capital "A".
            "description": f"Class for {class_name}",
            "properties": properties,
        }
        client.schema.create_class(class_obj)

    @classmethod
    @abstractmethod
    def _to_gpt_index(cls, client: Any, object_id: str, class_prefix: str) -> IS:
        """Convert to gpt index."""

    @classmethod
    def to_gpt_index(cls, client: Any, object_id: str, class_prefix: str) -> IS:
        """Convert to gpt index."""
        validate_client(client)
        result = cls._to_gpt_index(client, object_id, class_prefix)
        # Delete existing class_name, object_id from Weaviate.
        # NOTE: This is our current solution for preventing against duplicates.
        client.data_object.delete(object_id, cls._class_name(class_prefix))
        return result

    @classmethod
    @abstractmethod
    def _from_gpt_index(cls, client: Any, index: IS, class_prefix: str) -> str:
        """Convert from gpt index."""

    @classmethod
    def from_gpt_index(cls, client: Any, index: IS, class_prefix: str) -> str:
        """Convert from gpt index."""
        validate_client(client)
        # Create schema if doesn't exist
        cls.create_schema(client, class_prefix)
        index_id = cls._from_gpt_index(client, index, class_prefix)
        client.batch.flush()
        return index_id


class WeaviateNode(BaseWeaviateIndexStruct[Node]):
    """Weaviate node."""

    @classmethod
    def _class_name(cls, class_prefix: str) -> str:
        """Return class name."""
        return f"{class_prefix}_Node"

    @classmethod
    def _get_properties(cls) -> List[Dict]:
        """Create schema."""
        return [
            {
                "dataType": ["int"],
                "description": "The index of the Node",
                "name": "index",
            },
            {
                "dataType": ["int[]"],
                "description": "The child_indices of the Node",
                "name": "child_indices",
            },
            {
                "dataType": ["string"],
                "description": "The ref_doc_id of the Node",
                "name": "ref_doc_id",
            },
        ]

    @classmethod
    def _to_gpt_index(cls, client: Any, object_id: str, class_prefix: str) -> Node:
        """Convert to gpt index."""
        entry = cls._get_by_id(client, object_id, class_prefix)
        return Node(
            text=entry["text"],
            doc_id=entry["doc_id"],
            index=int(entry["index"]),
            child_indices=entry["child_indices"],
            ref_doc_id=entry["ref_doc_id"],
            embedding=entry["_additional"]["vector"],
        )

    @classmethod
    def _from_gpt_index(cls, client: Any, node: Node, class_prefix: str) -> str:
        """Convert from gpt index."""
        node_dict = node.to_dict()
        vector = node_dict.pop("embedding")
        # TODO: account for existing nodes that are stored
        node_id = get_new_id(set())
        class_name = cls._class_name(class_prefix)
        client.batch.add_data_object(node_dict, class_name, node_id, vector)

        return node_id


class WeaviateIndexGraph(BaseWeaviateIndexStruct[IndexGraph]):
    """Weaviate index graph."""

    @classmethod
    def _class_name(cls, class_prefix: str) -> str:
        """Return class name."""
        return f"{class_prefix}_IndexGraph"

    @classmethod
    def _get_properties(cls) -> List[Dict]:
        """Create schema."""
        return [
            {
                "dataType": ["string"],
                "description": "JSON-serialized all_nodes string of the IndexGraph",
                "name": "all_nodes",
            },
            {
                "dataType": ["string"],
                "description": "JSON-serialized root_nodes string of the IndexGraph",
                "name": "root_nodes",
            },
        ]

    @classmethod
    def _to_gpt_index(
        cls, client: Any, object_id: str, class_prefix: str
    ) -> IndexGraph:
        """Fetch stored object, convert to gpt index."""

        # load from weaviate
        entry = cls._get_by_id(client, object_id, class_prefix)
        weaviate_all_nodes = json.loads(entry["all_nodes"])
        weaviate_root_nodes = json.loads(entry["root_nodes"])

        # convert to gpt index
        all_nodes = {}
        for node_idx, node_id in weaviate_all_nodes.items():
            node: Node = WeaviateNode.to_gpt_index(client, node_id, class_prefix)
            all_nodes[node_idx] = node
        root_nodes = {}
        for node_idx, node_id in weaviate_root_nodes.items():
            node = WeaviateNode.to_gpt_index(client, node_id, class_prefix)
            root_nodes[node_idx] = node
        return IndexGraph(all_nodes=all_nodes, root_nodes=root_nodes)

    @classmethod
    def _from_gpt_index(
        cls,
        client: Any,
        index_graph: IndexGraph,
        class_prefix: str,
    ) -> str:
        """Convert from gpt index."""
        # first serialize/store nodes
        all_nodes_dict = {}
        for node_idx, node in index_graph.all_nodes.items():
            weaviate_node_id = WeaviateNode.from_gpt_index(client, node, class_prefix)
            all_nodes_dict[node_idx] = weaviate_node_id

        root_nodes_dict = {}
        for node_idx, node in index_graph.root_nodes.items():
            weaviate_node_id = WeaviateNode.from_gpt_index(client, node, class_prefix)
            root_nodes_dict[node_idx] = weaviate_node_id

        # then serialize/store index graph
        all_nodes_str = json.dumps(all_nodes_dict)
        root_nodes_str = json.dumps(root_nodes_dict)
        weaviate_index_graph_dict = {
            "all_nodes": all_nodes_str,
            "root_nodes": root_nodes_str,
        }

        class_name = cls._class_name(class_prefix)
        node_id = get_new_id(set())
        client.batch.add_data_object(weaviate_index_graph_dict, class_name, node_id)

        return node_id


_INDEX_STRUCT_TO_WEAVIATE_MAP: Dict[
    Type[IndexStruct], Type[BaseWeaviateIndexStruct]
] = {
    IndexGraph: WeaviateIndexGraph,
    Node: WeaviateNode,
}


def serialize_index_struct(
    index_struct: IndexStruct, client: Any, class_prefix: Optional[str] = None
) -> str:
    """Serialize index struct."""
    if type(index_struct) not in _INDEX_STRUCT_TO_WEAVIATE_MAP:
        raise ValueError(f"Cannot serialize index struct of type {type(index_struct)}")
    weaviate_cls = _INDEX_STRUCT_TO_WEAVIATE_MAP[type(index_struct)]
    class_prefix = class_prefix or DEFAULT_CLASS_PREFIX
    return weaviate_cls.from_gpt_index(client, index_struct, class_prefix)


def load_index_struct(
    index_struct_cls: Type[IndexStruct],
    client: Any,
    object_id: str,
    class_prefix: Optional[str] = None,
) -> IndexStruct:
    """Load index struct."""
    if index_struct_cls not in _INDEX_STRUCT_TO_WEAVIATE_MAP:
        raise ValueError(f"Cannot serialize index struct of type {index_struct_cls}")
    weaviate_cls = _INDEX_STRUCT_TO_WEAVIATE_MAP[index_struct_cls]
    class_prefix = class_prefix or DEFAULT_CLASS_PREFIX
    index_struct = weaviate_cls.to_gpt_index(client, object_id, class_prefix)
    return index_struct
