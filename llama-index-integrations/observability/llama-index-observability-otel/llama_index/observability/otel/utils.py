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
