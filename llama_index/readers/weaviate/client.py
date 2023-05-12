"""Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
import logging
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.data_structs.data_structs import Node
from llama_index.data_structs.node import DocumentRelationship
from llama_index.readers.weaviate.utils import parse_get_response, validate_client
from llama_index.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

_logger = logging.getLogger(__name__)

NODE_SCHEMA: List[Dict] = [
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


def create_schema(client: Any, class_prefix: str) -> None:
    """Create schema."""
    validate_client(client)
    # first check if schema exists
    schema = client.schema.get()
    classes = schema["classes"]
    existing_class_names = {c["class"] for c in classes}
    # if schema already exists, don't create
    class_name = _class_name(class_prefix)
    if class_name in existing_class_names:
        return

    properties = NODE_SCHEMA
    class_obj = {
        "class": _class_name(class_prefix),  # <= note the capital "A".
        "description": f"Class for {class_name}",
        "properties": properties,
    }
    client.schema.create_class(class_obj)


def weaviate_query(
    client: Any,
    class_prefix: str,
    query_spec: VectorStoreQuery,
) -> List[Node]:
    """Convert to LlamaIndex list."""
    validate_client(client)

    class_name = _class_name(class_prefix)
    prop_names = [p["name"] for p in NODE_SCHEMA]
    vector = query_spec.query_embedding

    # build query
    query = client.query.get(class_name, prop_names).with_additional(["id", "vector"])
    if query_spec.mode == VectorStoreQueryMode.DEFAULT:
        _logger.debug("Using vector search")
        if vector is not None:
            query = query.with_near_vector(
                {
                    "vector": vector,
                }
            )
    elif query_spec.mode == VectorStoreQueryMode.HYBRID:
        _logger.debug(f"Using hybrid search with alpha {query_spec.alpha}")
        query = query.with_hybrid(
            query=query_spec.query_str,
            alpha=query_spec.alpha,
            vector=vector,
        )
    query = query.with_limit(query_spec.similarity_top_k)
    _logger.debug(f"Using limit of {query_spec.similarity_top_k}")

    # execute query
    query_result = query.do()

    # parse results
    parsed_result = parse_get_response(query_result)
    entries = parsed_result[class_name]
    results = [_to_node(entry) for entry in entries]
    return results


def _class_name(class_prefix: str) -> str:
    """Return class name."""
    return f"{class_prefix}_Node"


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


def _to_node(entry: Dict) -> Node:
    """Convert to Node."""
    additional = entry.pop("_additional")
    try:
        extra_info, node_info, relationships = metadata_dict_to_node(entry)
    except Exception as e:
        _logger.debug("Failed to parse Node metadata, fallback to legacy logic.", e)
        extra_info, node_info, relationships = _legacy_metadata_dict_to_node(entry)

    return Node(
        text=entry["text"],
        embedding=additional["vector"],
        doc_id=entry["doc_id"],
        extra_info=extra_info,
        node_info=node_info,
        relationships=relationships,
    )


def _add_node(
    client: Any, node: Node, class_prefix: str, batch: Optional[Any] = None
) -> str:
    """Add node."""
    metadata = {}
    metadata["text"] = node.text or ""

    additional_metadata = node_to_metadata_dict(node)
    metadata.update(additional_metadata)

    # NOTE: important to set this after additional_metadata to override.
    #       be default, "doc_id" refers to source doc id, but for legacy reason
    #       we use "doc_id" to refer to node id in weaviate.
    metadata["doc_id"] = node.get_doc_id()

    vector = node.embedding
    node_id = node.get_doc_id()
    class_name = _class_name(class_prefix)

    # if batch object is provided (via a context manager), use that instead
    if batch is not None:
        batch.add_data_object(metadata, class_name, node_id, vector)
    else:
        client.batch.add_data_object(metadata, class_name, node_id, vector)

    return node_id


def delete_document(client: Any, ref_doc_id: str, class_prefix: str) -> None:
    """Delete entry."""
    validate_client(client)
    # make sure that each entry
    class_name = _class_name(class_prefix)
    where_filter = {
        "path": ["ref_doc_id"],
        "operator": "Equal",
        "valueString": ref_doc_id,
    }
    query = (
        client.query.get(class_name).with_additional(["id"]).with_where(where_filter)
    )

    query_result = query.do()
    parsed_result = parse_get_response(query_result)
    entries = parsed_result[class_name]
    for entry in entries:
        client.data_object.delete(entry["_additional"]["id"], class_name)


def add_node(client: Any, node: Node, class_prefix: str) -> str:
    """Convert from LlamaIndex."""
    validate_client(client)
    index_id = _add_node(client, node, class_prefix)
    client.batch.flush()
    return index_id


def add_nodes(client: Any, nodes: List[Node], class_prefix: str) -> List[str]:
    """Add nodes."""
    from weaviate import Client  # noqa: F401

    client = cast(Client, client)
    validate_client(client)
    index_ids = []
    with client.batch as batch:
        for node in nodes:
            index_id = _add_node(client, node, class_prefix, batch=batch)
            index_ids.append(index_id)
    return index_ids
