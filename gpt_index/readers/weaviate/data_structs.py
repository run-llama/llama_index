"""Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, cast

from gpt_index.data_structs.data_structs import IndexStruct, Node
from gpt_index.readers.weaviate.utils import (
    get_by_id,
    parse_get_response,
    validate_client,
)
from gpt_index.utils import get_new_id

IS = TypeVar("IS", bound=IndexStruct)


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
                "description": "Text property",
                "name": "text",
            },
            {
                "dataType": ["string"],
                "description": "Document id",
                "name": "doc_id",
            },
            {
                "dataType": ["string"],
                "description": "extra_info (in JSON)",
                "name": "extra_info",
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
    def _entry_to_gpt_index(cls, entry: Dict) -> IS:
        """Convert to LlamaIndex list."""

    @classmethod
    def to_gpt_index_list(
        cls,
        client: Any,
        class_prefix: str,
        vector: Optional[List[float]] = None,
        object_limit: Optional[int] = None,
    ) -> List[IS]:
        """Convert to LlamaIndex list."""
        validate_client(client)
        class_name = cls._class_name(class_prefix)
        properties = cls._get_common_properties() + cls._get_properties()
        prop_names = [p["name"] for p in properties]
        query = client.query.get(class_name, prop_names).with_additional(
            ["id", "vector"]
        )
        if vector is not None:
            query = query.with_near_vector(
                {
                    "vector": vector,
                }
            )
        if object_limit is not None:
            query = query.with_limit(object_limit)
        query_result = query.do()
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[class_name]

        results: List[IS] = []
        for entry in entries:
            results.append(cls._entry_to_gpt_index(entry))

        return results

    @classmethod
    @abstractmethod
    def _from_gpt_index(
        cls, client: Any, index: IS, class_prefix: str, batch: Optional[Any] = None
    ) -> str:
        """Convert from LlamaIndex."""

    @classmethod
    def from_gpt_index(cls, client: Any, index: IS, class_prefix: str) -> str:
        """Convert from LlamaIndex."""
        validate_client(client)
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
            {
                "dataType": ["string"],
                "description": "node_info (in JSON)",
                "name": "node_info",
            },
        ]

    @classmethod
    def _entry_to_gpt_index(cls, entry: Dict) -> Node:
        """Convert to LlamaIndex list."""
        extra_info_str = entry["extra_info"]
        if extra_info_str == "":
            extra_info = None
        else:
            extra_info = json.loads(extra_info_str)

        node_info_str = entry["node_info"]
        if node_info_str == "":
            node_info = None
        else:
            node_info = json.loads(node_info_str)
        return Node(
            text=entry["text"],
            doc_id=entry["doc_id"],
            index=int(entry["index"]),
            child_indices=entry["child_indices"],
            ref_doc_id=entry["ref_doc_id"],
            embedding=entry["_additional"]["vector"],
            extra_info=extra_info,
            node_info=node_info,
        )

    @classmethod
    def _from_gpt_index(
        cls, client: Any, node: Node, class_prefix: str, batch: Optional[Any] = None
    ) -> str:
        """Convert from LlamaIndex."""
        node_dict = node.to_dict()
        vector = node_dict.pop("embedding")
        extra_info = node_dict.pop("extra_info")
        # json-serialize the extra_info
        extra_info_str = ""
        if extra_info is not None:
            extra_info_str = json.dumps(extra_info)
        node_dict["extra_info"] = extra_info_str
        # json-serialize the node_info
        node_info = node_dict.pop("node_info")
        node_info_str = ""
        if node_info is not None:
            node_info_str = json.dumps(node_info)
        node_dict["node_info"] = node_info_str

        # TODO: account for existing nodes that are stored
        node_id = get_new_id(set())
        class_name = cls._class_name(class_prefix)

        # if batch object is provided (via a contexxt manager), use that instead
        if batch is not None:
            batch.add_data_object(node_dict, class_name, node_id, vector)
        else:
            client.batch.add_data_object(node_dict, class_name, node_id, vector)

        return node_id

    @classmethod
    def delete_document(cls, client: Any, ref_doc_id: str, class_prefix: str) -> None:
        """Delete entry."""
        validate_client(client)
        # make sure that each entry
        class_name = cls._class_name(class_prefix)
        where_filter = {
            "path": ["ref_doc_id"],
            "operator": "Equal",
            "valueString": ref_doc_id,
        }
        query = (
            client.query.get(class_name)
            .with_additional(["id"])
            .with_where(where_filter)
        )

        query_result = query.do()
        parsed_result = parse_get_response(query_result)
        entries = parsed_result[class_name]
        for entry in entries:
            client.data_object.delete(entry["_additional"]["id"], class_name)

    @classmethod
    def from_gpt_index_batch(
        cls, client: Any, nodes: List[Node], class_prefix: str
    ) -> List[str]:
        """Convert from gpt index."""
        from weaviate import Client  # noqa: F401

        client = cast(Client, client)
        validate_client(client)
        index_ids = []
        with client.batch as batch:
            for node in nodes:
                index_id = cls._from_gpt_index(client, node, class_prefix, batch=batch)
        index_ids.append(index_id)
        return index_ids
