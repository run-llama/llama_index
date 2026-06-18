"""
Borrowed from Langchain's Neo4j graph utility functions.

https://github.com/langchain-ai/langchain/blob/95c3e5f85f8ed8026a11e351b57bfae488d654c4/libs/community/langchain_community/graphs/neo4j_graph.py
"""

import logging
from typing import Any

LIST_LIMIT = 128

_logger = logging.getLogger(__name__)


def clean_string_values(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


def value_sanitize(d: Any) -> Any:
    """
    Sanitize the input dictionary or list.

    Sanitizes the input by removing embedding-like values,
    lists with more than 128 elements, that are mostly irrelevant for
    generating answers in a LLM context. These properties, if left in
    results, can occupy significant context space and detract from
    the LLM's performance by introducing unnecessary noise and cost.

    Pathologically nested inputs are dropped (treated the same as the
    function's existing "irrelevant value" branches) rather than allowed
    to blow the Python stack with a raw RecursionError.
    """
    try:
        return _value_sanitize_impl(d)
    except RecursionError:
        _logger.warning(
            "Graph value is too deeply nested to sanitize; dropping the value."
        )
        return None


def _value_sanitize_impl(d: Any) -> Any:
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sanitized_value = _value_sanitize_impl(value)
                if (
                    sanitized_value is not None
                ):  # Check if the sanitized value is not None
                    new_dict[key] = sanitized_value
            elif isinstance(value, list):
                if len(value) < LIST_LIMIT:
                    sanitized_value = _value_sanitize_impl(value)
                    if (
                        sanitized_value is not None
                    ):  # Check if the sanitized value is not None
                        new_dict[key] = sanitized_value
                # Do not include the key if the list is oversized
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) < LIST_LIMIT:
            return [
                _value_sanitize_impl(item)
                for item in d
                if _value_sanitize_impl(item) is not None
            ]
        else:
            return None
    else:
        return d
