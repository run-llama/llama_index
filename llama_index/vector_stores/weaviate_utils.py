"""Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
import logging
from dataclasses import field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from weaviate import Client

from llama_index.data_structs.data_structs import Node
from llama_index.data_structs.node import DocumentRelationship
from llama_index.vector_stores.utils import (DEFAULT_TEXT_KEY,
                                             metadata_dict_to_node,
                                             node_to_metadata_dict)

_logger = logging.getLogger(__name__)

NODE_SCHEMA: List[Dict] = [
    {
        "dataType": ["string"],
        "description": "Text property",
        "name": "text",
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
    {
        "dataType": ["string"],
        "description": "The relationships of the node (in JSON)",
        "name": "relationships",
    },
]


def validate_client(client: Any) -> None:
    """Validate client and import weaviate library."""
    try:
        import weaviate  # noqa: F401
        from weaviate import Client

        client = cast(Client, client)
    except ImportError:
        raise ImportError(
            "Weaviate is not installed. "
            "Please install it with `pip install weaviate-client`."
        )
    cast(Client, client)


def parse_get_response(response: Dict) -> Dict:
    """Parse get response from Weaviate."""
    if "errors" in response:
        raise ValueError("Invalid query, got errors: {}".format(response["errors"]))
    data_response = response["data"]
    if "Get" not in data_response:
        raise ValueError("Invalid query response, must be a Get query.")

    return data_response["Get"]


def class_schema_exists(client: Any, class_name: str) -> None:
    """Create schema."""
    validate_client(client)
    schema = client.schema.get()
    classes = schema["classes"]
    existing_class_names = {c["class"] for c in classes}
    return class_name in existing_class_names


def create_default_schema(client: Any, class_name: str) -> None:
    """Create default schema."""
    validate_client(client)
    class_schema = {
        "class": class_name,
        "description": f"Class for {class_name}",
        "properties": NODE_SCHEMA,
    }
    client.schema.create_class(class_schema)


def get_all_properties(client: Any, class_name: str) -> None:
    """Get all properties of a class."""
    validate_client(client)
    schema = client.schema.get()
    classes = schema["classes"]
    classes_by_name = {c["class"]: c for c in classes}
    if class_name not in classes_by_name:
        raise ValueError(f'{class_name} schema does not exist.')
    schema = classes_by_name[class_name]
    return [p['name'] for p in schema['properties']]


def _legacy_metadata_dict_to_node(entry: Dict[str, Any]) -> Tuple[dict, dict, dict]:
    """Legacy logic for converting metadata dict to node data.
    Only for backwards compatibility.
    """
    extra_info_str = entry["extra_info"]
    if extra_info_str == "":
        extra_info = {}
    else:
        extra_info = json.loads(extra_info_str)

    node_info_str = entry["node_info"]
    if node_info_str == "":
        node_info = {}
    else:
        node_info = json.loads(node_info_str)

    relationships_str = entry["relationships"]
    relationships: Dict[DocumentRelationship, str]
    if relationships_str == "":
        relationships = field(default_factory=dict)
    else:
        relationships = {
            DocumentRelationship(k): v for k, v in json.loads(relationships_str).items()
        }
    return extra_info, node_info, relationships


def to_node(entry: Dict, text_key: str = DEFAULT_TEXT_KEY) -> Node:
    """Convert to Node."""
    additional = entry.pop("_additional")
    try:
        extra_info, node_info, relationships = metadata_dict_to_node(entry)
    except Exception as e:
        _logger.debug("Failed to parse Node metadata, fallback to legacy logic.", e)
        extra_info, node_info, relationships = _legacy_metadata_dict_to_node(entry)

    return Node(
        text=entry.get(text_key, ""),
        embedding=additional["vector"],
        doc_id=additional["id"],
        extra_info=extra_info,
        node_info=node_info,
        relationships=relationships,
    )


def add_node(
    client: "Client",
    node: Node,
    class_name: str,
    batch: Optional[Any] = None,
    text_key: str = DEFAULT_TEXT_KEY,
) -> None:
    """Add node."""
    metadata = {}
    metadata[text_key] = node.text or ""

    additional_metadata = node_to_metadata_dict(node)
    metadata.update(additional_metadata)

    vector = node.embedding
    id = node.get_doc_id()

    # if batch object is provided (via a context manager), use that instead
    if batch is not None:
        batch.add_data_object(metadata, class_name, id, vector)
    else:
        client.batch.add_data_object(metadata, class_name, id, vector)
