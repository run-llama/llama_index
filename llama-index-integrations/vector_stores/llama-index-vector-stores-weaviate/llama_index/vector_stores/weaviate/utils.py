"""
Weaviate-specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import logging
from typing import Any, Dict, List, Optional, cast


from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
import weaviate.classes as wvc

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
        import weaviate

        client = cast(weaviate.WeaviateClient, client)
    except ImportError:
        raise ImportError(
            "Weaviate is not installed. "
            "Please install it with `pip install weaviate-client`."
        )
    cast(weaviate.WeaviateClient, client)


def validate_async_client(client: Any) -> None:
    """Validate client and import weaviate library."""
    try:
        import weaviate

        client = cast(weaviate.WeaviateAsyncClient, client)
    except ImportError:
        raise ImportError(
            "Weaviate is not installed. "
            "Please install it with `pip install weaviate-client`."
        )
    cast(weaviate.WeaviateAsyncClient, client)


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
    return client.collections.exists(class_name)


async def aclass_schema_exists(client: Any, class_name: str) -> bool:
    """Check if class schema exists."""
    validate_async_client(client)
    collection = client.collections.get(class_name)
    return await collection.exists()


def create_default_schema(client: Any, class_name: str) -> None:
    """Create default schema."""
    validate_client(client)
    class_schema = {
        "class": class_name,
        "description": f"Class for {class_name}",
        "properties": NODE_SCHEMA,
    }
    client.collections.create_from_dict(class_schema)


async def acreate_default_schema(client: Any, class_name: str) -> None:
    """Create default schema."""
    validate_async_client(client)
    class_schema = {
        "class": class_name,
        "description": f"Class for {class_name}",
        "properties": NODE_SCHEMA,
    }
    await client.collections.create_from_dict(class_schema)


def get_node_similarity(entry: Dict, similarity_key: str = "score") -> float:
    """Get converted node similarity from distance."""
    score = getattr(entry["metadata"], similarity_key)

    if score is None:
        return 0.0

    # The hybrid search in Weaviate returns similarity score: https://weaviate.io/developers/weaviate/search/hybrid#explain-the-search-results
    return float(score)


def to_node(entry: Dict, text_key: str = DEFAULT_TEXT_KEY) -> TextNode:
    """Convert to Node."""
    additional = entry["metadata"].__dict__
    text = entry["properties"].pop(text_key, "")

    embedding = entry["vector"].pop("default", None)

    try:
        node = metadata_dict_to_node(entry["properties"])
        node.text = text
        node.embedding = embedding
    except Exception as e:
        _logger.debug("Failed to parse Node metadata, fallback to legacy logic. %s", e)
        metadata, node_info, relationships = legacy_metadata_dict_to_node(entry)

        node = TextNode(
            text=text,
            id_=additional.get("id", str(metadata.get("uuid"))),
            metadata=metadata,
            start_char_idx=node_info.get("start", None),
            end_char_idx=node_info.get("end", None),
            relationships=relationships,
            embedding=embedding,
        )
    return node


def get_collection_vectorizer_config(
    client: Any, class_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get the vectorizer configuration from an existing Weaviate collection.

    Returns the vectorizer config if found, None otherwise.
    Vectorizers like 'text2vec-openai', 'text2vec-cohere', etc. are required
    for native_embedding feature to work.
    """
    validate_client(client)
    try:
        collection = client.collections.get(class_name)
        config = collection.config.get()
        if config and hasattr(config, "vectorizer") and config.vectorizer:
            return config.vectorizer
    except Exception as e:
        _logger.debug(f"Failed to retrieve vectorizer config for {class_name}: {e}")
    return None


async def aget_collection_vectorizer_config(
    client: Any, class_name: str
) -> Optional[Dict[str, Any]]:
    """
    Asynchronously get the vectorizer configuration from an existing Weaviate collection.

    Returns the vectorizer config if found, None otherwise.
    Vectorizers like 'text2vec-openai', 'text2vec-cohere', etc. are required
    for native_embedding feature to work.
    """
    validate_async_client(client)
    try:
        collection = client.collections.get(class_name)
        config = await collection.config.get()
        if config and hasattr(config, "vectorizer") and config.vectorizer:
            return config.vectorizer
    except Exception as e:
        _logger.debug(f"Failed to retrieve vectorizer config for {class_name}: {e}")
    return None


def get_data_object(
    node: BaseNode,
    text_key: str = DEFAULT_TEXT_KEY,
    use_vector: bool = True,
) -> dict:
    """Add node."""
    metadata = {}
    metadata[text_key] = node.get_content(metadata_mode=MetadataMode.NONE) or ""

    additional_metadata = node_to_metadata_dict(
        node, remove_text=True, flat_metadata=False
    )
    metadata.update(additional_metadata)

    vector = node.get_embedding() if use_vector else None
    id = node.node_id

    return wvc.data.DataObject(properties=metadata, uuid=id, vector=vector)
