import json
from collections.abc import Sequence as ABCSequence
from typing import Any

BASE_TYPES = (int, str, bool, bytes, float)


def _is_otel_supported_type(obj: Any) -> bool:
    # If it's one of the base types
    if isinstance(obj, BASE_TYPES):
        return True

    # If it's a sequence (but not a string or bytes, which are sequences too)
    if isinstance(obj, ABCSequence) and not isinstance(obj, (str, bytes)):
        return all(isinstance(item, BASE_TYPES) for item in obj)

    return False


def filter_model_fields(model_dict: dict) -> dict:
    newdict = {}
    for field in model_dict:
        if _is_otel_supported_type(model_dict[field]):
            newdict.update({field: model_dict[field]})

    return newdict


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary into a single-level dict with dot-notation keys.

    Nested dicts are recursively flattened. Values that are OTel-supported types
    are kept as-is. Unsupported types (e.g., nested lists of dicts) are JSON
    serialized to preserve the data.

    Example:
        {"user": {"name": "alice", "age": 30}}
        becomes
        {"user.name": "alice", "user.age": 30}

    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif _is_otel_supported_type(v):
            items.append((new_key, v))
        else:
            # Fallback: JSON serialize unsupported types to preserve data
            items.append((new_key, json.dumps(v, default=str)))
    return dict(items)
