import json
from typing import Any, Dict

from llama_index.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
    TextNode,
)

DEFAULT_TEXT_KEY = "text"
DEFAULT_EMBEDDING_KEY = "embedding"
DEFAULT_DOC_ID_KEY = "doc_id"


def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float, None).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float, type(None))):
            raise ValueError(
                f"Value for metadata {key} must be one of (str, int, float, None)"
            )


def node_to_metadata_dict(
    node: BaseNode,
    remove_text: bool = False,
    text_field: str = DEFAULT_TEXT_KEY,
    flat_metadata: bool = False,
) -> Dict[str, Any]:
    """Common logic for saving Node data into metadata dict."""
    node_dict = node.dict()
    metadata: Dict[str, Any] = node_dict.get("metadata", {})

    if flat_metadata:
        _validate_is_flat_dict(metadata)

    # store entire node as json string - some minor text duplication
    if remove_text:
        node_dict[text_field] = ""

    # remove embedding from node_dict
    node_dict["embedding"] = None

    # dump remainder of node_dict to json string
    metadata["_node_content"] = json.dumps(node_dict)
    metadata["_node_type"] = node.class_name()

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

    return metadata


def metadata_dict_to_node(metadata: dict) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content", None)
    node_type = metadata.get("_node_type", None)
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    if node_type == IndexNode.class_name():
        return IndexNode.parse_raw(node_json)
    elif node_type == ImageNode.class_name():
        return ImageNode.parse_raw(node_json)
    else:
        return TextNode.parse_raw(node_json)
