import json
from collections.abc import Sequence as ABCSequence
from typing import Any, Dict, Optional, Union

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


# OpenTelemetry GenAI semantic convention attribute names
# (https://github.com/open-telemetry/semantic-conventions-genai).
# Spelled out here because the constants in
# `opentelemetry.semconv._incubating` are marked deprecated (moved to the
# GenAI repo) and the cache one is missing from older releases we support.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"

# LLM start/end events from llama_index.core, matched by class name because
# this package only depends on llama-index-instrumentation.
_GEN_AI_START_EVENTS = {
    "LLMChatStartEvent": "chat",
    "LLMCompletionStartEvent": "text_completion",
}
_GEN_AI_END_EVENTS = {"LLMChatEndEvent", "LLMCompletionEndEvent"}


def _get(obj: Any, key: str) -> Any:
    """Read a key from a dict or an attribute from an object (e.g. a pydantic model)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(obj: Any, key: str) -> Optional[int]:
    value = _get(obj, key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _sum(*values: Optional[int]) -> Optional[int]:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def usage_to_gen_ai_attributes(usage: Any) -> Dict[str, int]:
    """
    Map a provider usage payload to GenAI token attributes.

    Handles the OpenAI chat completions shape (``prompt_tokens``), the Gemini
    shape (``prompt_token_count``) and the Anthropic / OpenAI Responses shape
    (``input_tokens``). Anthropic reports cache reads and writes separately
    from ``input_tokens``, so they are added back in, as the conventions ask.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read: Optional[int] = None
    if _int(usage, "prompt_tokens") is not None:
        input_tokens = _int(usage, "prompt_tokens")
        output_tokens = _int(usage, "completion_tokens")
        cache_read = _int(_get(usage, "prompt_tokens_details"), "cached_tokens")
    elif _int(usage, "prompt_token_count") is not None:
        input_tokens = _int(usage, "prompt_token_count")
        # Gemini reports thinking tokens apart from candidates; output tokens
        # should include reasoning tokens.
        output_tokens = _sum(
            _int(usage, "candidates_token_count"), _int(usage, "thoughts_token_count")
        )
        cache_read = _int(usage, "cached_content_token_count")
    elif _int(usage, "input_tokens") is not None:
        cache_read = _int(usage, "cache_read_input_tokens")
        input_tokens = _sum(
            _int(usage, "input_tokens"),
            cache_read,
            _int(usage, "cache_creation_input_tokens"),
        )
        output_tokens = _int(usage, "output_tokens")
        if cache_read is None:
            cache_read = _int(_get(usage, "input_tokens_details"), "cached_tokens")

    attributes: Dict[str, int] = {}
    if input_tokens is not None:
        attributes[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    if cache_read:
        attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
    return attributes


def gen_ai_attributes_from_event(event: Any) -> Dict[str, Union[str, int]]:
    """GenAI semantic convention span attributes for an LLM start or end event."""
    name = event.class_name()
    attributes: Dict[str, Union[str, int]] = {}
    if name in _GEN_AI_START_EVENTS:
        attributes[GEN_AI_OPERATION_NAME] = _GEN_AI_START_EVENTS[name]
        model_dict = getattr(event, "model_dict", None)
        model = _get(model_dict, "model") or _get(model_dict, "model_name")
        # "unknown" is the LLMMetadata placeholder, not a model name
        if isinstance(model, str) and model and model != "unknown":
            attributes[GEN_AI_REQUEST_MODEL] = model
    elif name in _GEN_AI_END_EVENTS:
        response = getattr(event, "response", None)
        raw = _get(response, "raw")
        additional_kwargs = _get(response, "additional_kwargs") or {}
        response_model = _get(raw, "model") or _get(raw, "model_version")
        if isinstance(response_model, str) and response_model:
            attributes[GEN_AI_RESPONSE_MODEL] = response_model
        for usage in (
            _get(raw, "usage"),
            _get(raw, "usage_metadata"),
            _get(additional_kwargs, "usage"),
            additional_kwargs,
        ):
            usage_attributes = usage_to_gen_ai_attributes(usage)
            if usage_attributes:
                attributes.update(usage_attributes)
                break
    return attributes
