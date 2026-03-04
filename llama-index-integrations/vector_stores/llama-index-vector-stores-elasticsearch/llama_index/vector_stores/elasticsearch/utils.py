from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch, Elasticsearch
from logging import getLogger

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node

logger = getLogger(__name__)


def get_user_agent() -> str:
    """Get user agent for Elasticsearch client."""
    import llama_index.core

    version = getattr(llama_index.core, "__version__", "")
    return f"llama_index-py-vs/{version}"


def get_elasticsearch_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> AsyncElasticsearch:
    if url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if url:
        connection_params["hosts"] = [url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    sync_es_client = Elasticsearch(
        **connection_params, headers={"user-agent": get_user_agent()}
    )
    async_es_client = AsyncElasticsearch(
        **connection_params, headers={"user-agent": get_user_agent()}
    )

    sync_es_client.info()  # use sync client so don't have to 'await' to just get info

    return async_es_client


def convert_es_hit_to_node(
    hit: Dict[str, Any], text_field: str = "content"
) -> BaseNode:
    """
    Convert an Elasticsearch search hit to a BaseNode.

    Args:
        hit: The Elasticsearch search hit
        text_field: The field name that contains the text content

    Returns:
        BaseNode: The converted node

    """
    source = hit.get("_source", {})
    metadata = source.get("metadata", {})
    text = source.get(text_field, None)
    node_id = hit.get("_id")

    try:
        node = metadata_dict_to_node(metadata)
        node.text = text
    except Exception:
        # Legacy support for old metadata format
        logger.warning(f"Could not parse metadata from hit {source.get('metadata')}")
        node_info = source.get("node_info")
        relationships = source.get("relationships", {})
        start_char_idx = None
        end_char_idx = None
        if isinstance(node_info, dict):
            start_char_idx = node_info.get("start", None)
            end_char_idx = node_info.get("end", None)

        node = TextNode(
            text=text,
            metadata=metadata,
            id_=node_id,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            relationships=relationships,
        )

    return node
