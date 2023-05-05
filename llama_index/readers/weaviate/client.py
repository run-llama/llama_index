"""Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
from dataclasses import field
from typing import Any, Dict, List, Optional, cast

from llama_index.data_structs.data_structs import Node
from llama_index.data_structs.node import DocumentRelationship
from llama_index.readers.weaviate.utils import (
    get_by_id,
    parse_get_response,
    validate_client,
)
from llama_index.utils import get_new_id
from llama_index.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

import logging

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
        "description": "extra_info (in JSON)",
        "name": "extra_info",
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
        "description": "The hash of the Document",
        "name": "doc_hash",
    },
    {
        "dataType": ["string"],
        "description": "The relationships of the node (in JSON)",
        "name": "relationships",
    },
]


def _get_by_id(client: Any, object_id: str, class_prefix: str) -> Dict:
    """Get entry by id."""
    validate_client(client)
    class_name = _class_name(class_prefix)
    properties = NODE_SCHEMA
    prop_names = [p["name"] for p in properties]
    entry = get_by_id(client, object_id, class_name, prop_names)
    return entry


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


def _to_node(entry: Dict) -> Node:
    """Convert to Node."""
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

    relationships_str = entry["relationships"]
    relationships: Dict[DocumentRelationship, str]
    if relationships_str == "":
        relationships = field(default_factory=dict)
    else:
        relationships = {
            DocumentRelationship(k): v for k, v in json.loads(relationships_str).items()
        }

    return Node(
        text=entry["text"],
        doc_id=entry["doc_id"],
        embedding=entry["_additional"]["vector"],
        extra_info=extra_info,
        node_info=node_info,
        relationships=relationships,
    )


def _node_to_dict(node: Node) -> dict:
    node_dict = node.to_dict()
    node_dict.pop("embedding")  # NOTE: stored outside of dict

    # json-serialize the extra_info
    extra_info = node_dict.pop("extra_info")
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

    # json-serialize the relationships
    relationships = node_dict.pop("relationships")
    relationships_str = ""
    if relationships is not None:
        relationships_str = json.dumps(relationships)
    node_dict["relationships"] = relationships_str

    ref_doc_id = node.ref_doc_id
    if ref_doc_id is not None:
        node_dict["ref_doc_id"] = ref_doc_id
    return node_dict


def _add_node(
    client: Any, node: Node, class_prefix: str, batch: Optional[Any] = None
) -> str:
    """Add node."""
    node_dict = _node_to_dict(node)
    vector = node.embedding

    # TODO: account for existing nodes that are stored
    node_id = get_new_id(set())
    class_name = _class_name(class_prefix)

    # if batch object is provided (via a context manager), use that instead
    if batch is not None:
        batch.add_data_object(node_dict, class_name, node_id, vector)
    else:
        client.batch.add_data_object(node_dict, class_name, node_id, vector)

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
