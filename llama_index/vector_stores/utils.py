import json
from typing import Any, Dict, Tuple

from llama_index.schema import (
    BaseNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
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
    metadata["_node_type"] = node.get_type()

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

    if node_type == IndexNode.get_type():
        return IndexNode.parse_raw(node_json)
    else:
        return TextNode.parse_raw(node_json)


# TODO: Deprecated conversion functions
def legacy_metadata_dict_to_node(
    metadata: dict, text_key: str = DEFAULT_TEXT_KEY
) -> Tuple[dict, dict, dict]:
    """Common logic for loading Node data from metadata dict."""
    # make a copy first
    if metadata is None:
        metadata = {}
    else:
        metadata = metadata.copy()

    # load node_info from json string
    node_info_str = metadata.pop("node_info", "")
    if node_info_str == "":
        node_info = {}
    else:
        node_info = json.loads(node_info_str)

    # load relationships from json string
    relationships_str = metadata.pop("relationships", "")
    relationships: Dict[NodeRelationship, RelatedNodeInfo]
    if relationships_str == "":
        relationships = {}
    else:
        relationships = {
            NodeRelationship(k): RelatedNodeInfo(node_id=str(v))
            for k, v in json.loads(relationships_str).items()
        }

    # remove other known fields
    metadata.pop(text_key, None)
    metadata.pop("id", None)
    metadata.pop("document_id", None)
    metadata.pop("doc_id", None)
    metadata.pop("ref_doc_id", None)

    # remaining metadata is metadata or node_info
    new_metadata = {}
    for key, val in metadata.items():
        # NOTE: right now we enforce metadata to be dict of simple types.
        #       dump anything that's not a simple type into node_info.
        if isinstance(val, (str, int, float, type(None))):
            new_metadata[key] = val
        else:
            node_info[key] = val

    return new_metadata, node_info, relationships
