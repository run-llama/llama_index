import json
from typing import Any, Dict, Tuple

from llama_index.schema import BaseNode, NodeRelationship, RelatedNodeInfo

DEFAULT_TEXT_KEY = "text"


def node_to_metadata_dict(node: BaseNode) -> dict:
    """Common logic for saving Node data into metadata dict."""
    metadata: Dict[str, Any] = {}

    # store extra_info directly to allow metadata filtering
    if node.metadata is not None:
        metadata.update(node.metadata)

    # json-serialize the node_info
    node_info_str = ""
    if node.node_info is not None:  # type: ignore[attr-defined]
        node_info_str = json.dumps(node.node_info)  # type: ignore[attr-defined]
    metadata["node_info"] = node_info_str

    # json-serialize the relationships
    relationships_str = ""
    if node.relationships is not None:
        relationships_str = json.dumps(node.relationships)
    metadata["relationships"] = relationships_str

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

    return metadata


def metadata_dict_to_node(
    metadata: dict, text_key: str = DEFAULT_TEXT_KEY
) -> Tuple[dict, dict, dict]:
    """Common logic for loading Node data from metadata dict."""
    # make a copy first
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
            NodeRelationship(k): v for k, v in json.loads(relationships_str).items()
        }

    # remove other known fields
    metadata.pop(text_key, None)
    metadata.pop("id", None)
    metadata.pop("document_id", None)
    metadata.pop("doc_id", None)
    metadata.pop("ref_doc_id", None)

    # remaining metadata is extra_info or node_info
    extra_info = {}
    for key, val in metadata.items():
        # NOTE: right now we enforce extra_info to be dict of simple types.
        #       dump anything that's not a simple type into node_info.
        if isinstance(val, (str, int, float, type(None))):
            extra_info[key] = val
        else:
            node_info[key] = val

    return extra_info, node_info, relationships
