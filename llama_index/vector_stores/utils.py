from typing import Any, Dict
from llama_index.data_structs.node import Node


def get_metadata_from_node_info(
    node_info: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get metadata from node extra info."""
    metadata = {}
    for key, value in node_info.items():
        metadata[field_prefix + "_" + key] = value
    return metadata


def get_node_info_from_metadata(
    metadata: Dict[str, Any], field_prefix: str
) -> Dict[str, Any]:
    """Get node extra info from metadata."""
    node_extra_info = {}
    for key, value in metadata.items():
        if key.startswith(field_prefix + "_"):
            node_extra_info[key.replace(field_prefix + "_", "")] = value
    return node_extra_info


def node_to_metadata_dict(node: Node):
    metadata = {
        "text": node.get_text(),
        "doc_id": node.ref_doc_id,
        "id": node.get_doc_id(),
    }

    if node.extra_info:
        metadata.update(get_metadata_from_node_info(node.extra_info, "extra_info"))
    if node.node_info:
        metadata.update(get_metadata_from_node_info(node.node_info, "node_info"))
    if node.relationships:
        metadata.update(
            get_metadata_from_node_info(node.relationships, "relationships")
        )

    return metadata


def metadata_dict_to_node(metadata: dict) -> Node:
    text = metadata["text"]
    extra_info = get_node_info_from_metadata(metadata, "extra_info")
    node_info = get_node_info_from_metadata(metadata, "node_info")
    relationships = get_node_info_from_metadata(metadata, "relationships")
    doc_id = metadata["id"]

    return Node(
        text=text,
        extra_info=extra_info,
        node_info=node_info,
        doc_id=doc_id,
        relationships=relationships,
    )
