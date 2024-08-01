import json
from typing import Any, Dict, Optional, Tuple

from llama_index.legacy.schema import (
    BaseNode,
    ImageNode,
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
    metadata["_node_type"] = node.class_name()

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

    return metadata


def metadata_dict_to_node(metadata: dict, text: Optional[str] = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content", None)
    node_type = metadata.get("_node_type", None)
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.parse_raw(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.parse_raw(node_json)
    else:
        node = TextNode.parse_raw(node_json)

    if text is not None:
        node.set_content(text)

    return node


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

    id_ = metadata.pop("id", None)
    document_id = metadata.pop("document_id", None)
    doc_id = metadata.pop("doc_id", None)
    ref_doc_id = metadata.pop("ref_doc_id", None)

    # don't remove id's from metadata that llama-index doesn't know about
    ref_doc_id_info = relationships.get(NodeRelationship.PARENT, None)
    if ref_doc_id_info is not None:
        ref_doc_id = ref_doc_id_info.node_id

    if id_ is not None and id_ != ref_doc_id:
        metadata["id"] = id_
    if document_id is not None and document_id != ref_doc_id:
        metadata["document_id"] = document_id
    if doc_id is not None and doc_id != ref_doc_id:
        metadata["doc_id"] = doc_id

    # remaining metadata is metadata or node_info
    new_metadata = {}
    for key, val in metadata.items():
        # don't enforce types on metadata anymore (we did in the past)
        # since how we store this data now has been updated
        new_metadata[key] = val

    return new_metadata, node_info, relationships
