from typing import Any, Dict, Optional, Type
from gpt_index.constants import VECTOR_STORE_KEY
from gpt_index.vector_stores.registry import (
    VectorStoreType,
    load_vector_store_from_dict,
    save_vector_store_to_dict,
)
from gpt_index.vector_stores.types import VectorStore


def save_query_context_to_dict(
    query_context: Dict[str, Dict[str, Any]],
    vector_store_cls_to_type: Optional[Dict[Type[VectorStore], VectorStoreType]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Save index-specific query context dict to JSON dict.

    Example query context dict to save:
    query_context = {
        <index_id>: {
            'vector_store': <vector_store>
        }
    }

    NOTE: Right now, only consider vector stores.
    """
    save_dict = {}
    for index_id, index_context_dict in query_context.items():
        index_save_dict = {}
        for key, val in index_context_dict.items():
            if isinstance(val, VectorStore):
                index_save_dict[key] = save_vector_store_to_dict(
                    val, cls_to_type=vector_store_cls_to_type
                )
        save_dict[index_id] = index_save_dict
    return save_dict


def load_query_context_from_dict(
    save_dict: Dict[str, Dict[str, Any]],
    vector_store_type_to_cls: Optional[Dict[VectorStoreType, Type[VectorStore]]] = None,
    query_context_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load index-specific query context from JSON dict.

    Example loaded query context dict:
    query_context = {
        <index_id>: {
            'vector_store': <vector_store>
        }
    }

    NOTE: Right now, only consider vector stores.
    """
    if query_context_kwargs is None:
        query_context_kwargs = {}

    context_dict = {}
    for index_id, index_save_dict in save_dict.items():
        index_context_dict = {}
        index_kwargs = query_context_kwargs.get(index_id, {})
        for key, val in index_save_dict.items():
            if key == VECTOR_STORE_KEY:
                key_kwargs = index_kwargs.get(key, {})
                index_context_dict[key] = load_vector_store_from_dict(
                    val, vector_store_type_to_cls, **key_kwargs
                )
            else:
                index_context_dict[key] = val
        context_dict[index_id] = index_context_dict
    return context_dict
