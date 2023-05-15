import json
from dataclasses import field
from typing import Any, Dict, Tuple

from llama_index.data_structs.node import DocumentRelationship, Node


def node_to_metadata_dict(node: Node) -> dict:
    """Common logic for saving Node data into metadata dict."""
    metadata: Dict[str, Any] = {}

    # store extra_info directly to allow metadata filtering
    if node.extra_info is not None:
        metadata.update(node.extra_info)

    # json-serialize the node_info
    node_info_str = ""
    if node.node_info is not None:
        node_info_str = json.dumps(node.node_info)
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


def metadata_dict_to_node(metadata: dict) -> Tuple[dict, dict, dict]:
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
    relationships: Dict[DocumentRelationship, str]
    if relationships_str == "":
        relationships = field(default_factory=dict)
    else:
        relationships = {
            DocumentRelationship(k): v for k, v in json.loads(relationships_str).items()
        }

    # remove other known fields
    metadata.pop("text", None)
    metadata.pop("id", None)
    metadata.pop("document_id", None)
    metadata.pop("doc_id", None)
    metadata.pop("ref_doc_id", None)

    # remaining metadata is extra_info
    extra_info = metadata

    return extra_info, node_info, relationships
