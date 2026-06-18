from typing import Any

from llama_index.core.bridge.pydantic import BaseModel


def maybe_model_dump(raw: Any) -> Any:
    """Best-effort serialization for raw provider responses."""
    try:
        is_pydantic_model = isinstance(raw, BaseModel)
    except Exception:
        return raw

    if not is_pydantic_model:
        return raw

    try:
        return raw.model_dump()
    except Exception:
        return raw
