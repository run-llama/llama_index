"""Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from weaviate import Client

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

NODE_SCHEMA: List[Dict] = [
    {
        "dataType": ["text"],
        "description": "Text property",
        "name": "text",
    },
    {
        "dataType": ["text"],
        "description": "The ref_doc_id of the Node",
        "name": "ref_doc_id",
    },
    {
        "dataType": ["text"],
        "description": "node_info (in JSON)",
        "name": "node_info",
    },
    {
        "dataType": ["text"],
        "description": "The relationships of the node (in JSON)",
        "name": "relationships",
    },
]


def validate_client(client: Any) -> None:
    """Validate client and import weaviate library."""
    try:
        import weaviate  # noqa
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


def class_schema_exists(client: Any, class_name: str) -> bool:
    """Check if class schema exists."""
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


def get_all_properties(client: Any, class_name: str) -> List[str]:
    """Get all properties of a class."""
    validate_client(client)
    schema = client.schema.get()
    classes = schema["classes"]
    classes_by_name = {c["class"]: c for c in classes}
    if class_name not in classes_by_name:
        raise ValueError(f"{class_name} schema does not exist.")
    schema = classes_by_name[class_name]
    return [p["name"] for p in schema["properties"]]


def get_node_similarity(entry: Dict, similarity_key: str = "distance") -> float:
    """Get converted node similarity from distance."""
    distance = entry["_additional"].get(similarity_key, 0.0)

    if distance is None:
        return 1.0

    # convert distance https://forum.weaviate.io/t/distance-vs-certainty-scores/258
    return 1.0 - float(distance)


def to_node(entry: Dict, text_key: str = DEFAULT_TEXT_KEY) -> TextNode:
    """Convert to Node."""
    additional = entry.pop("_additional")
    text = entry.pop(text_key, "")
    embedding = additional.pop("vector", None)
    try:
        node = metadata_dict_to_node(entry)
        node.text = text
        node.embedding = embedding
    except Exception as e:
        _logger.debug("Failed to parse Node metadata, fallback to legacy logic.", e)
        metadata, node_info, relationships = legacy_metadata_dict_to_node(entry)

        node = TextNode(
            text=text,
            id_=additional["id"],
            metadata=metadata,
            start_char_idx=node_info.get("start", None),
            end_char_idx=node_info.get("end", None),
            relationships=relationships,
            embedding=embedding,
        )
    return node


def add_node(
    client: "Client",
    node: BaseNode,
    class_name: str,
    batch: Optional[Any] = None,
    text_key: str = DEFAULT_TEXT_KEY,
) -> None:
    """Add node."""
    metadata = {}
    metadata[text_key] = node.get_content(metadata_mode=MetadataMode.NONE) or ""

    additional_metadata = node_to_metadata_dict(
        node, remove_text=True, flat_metadata=False
    )
    metadata.update(additional_metadata)

    vector = node.get_embedding()
    id = node.node_id

    # if batch object is provided (via a context manager), use that instead
    if batch is not None:
        batch.add_data_object(metadata, class_name, id, vector)
    else:
        client.batch.add_data_object(metadata, class_name, id, vector)
