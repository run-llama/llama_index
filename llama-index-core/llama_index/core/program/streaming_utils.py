"""
Simplified streaming utilities for processing structured outputs from message content.

This module provides utilities for processing streaming responses that contain
structured data directly in the message content (not in function calls).
"""

import json
from typing import Any, List, Optional, Tuple, Type, Union

from pydantic import ValidationError

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.program.utils import (
    FlexibleModel,
    _repair_incomplete_json,
    create_flexible_model,
)
from llama_index.core.types import Model


def process_streaming_content_incremental(
    chat_response: ChatResponse,
    output_cls: Type[Model],
    cur_object: Optional[Union[Model, FlexibleModel]] = None,
) -> Union[Model, FlexibleModel]:
    """
    Process streaming response content with true incremental list handling.

    This version can extract partial progress from incomplete JSON and build
    lists incrementally (e.g., 1 joke → 2 jokes → 3 jokes) rather than
    jumping from empty to complete lists.

    Args:
        chat_response (ChatResponse): The chat response to process
        output_cls (Type[BaseModel]): The target output class
        cur_object (Optional[BaseModel]): Current best object (for comparison)
        flexible_mode (bool): Whether to use flexible schema during parsing

    Returns:
        Union[BaseModel, FlexibleModel]: Processed object with incremental updates

    """
    partial_output_cls = create_flexible_model(output_cls)

    # Get content from message
    content = chat_response.message.content
    if not content:
        return cur_object if cur_object is not None else partial_output_cls()
    parsed_obj: Optional[FlexibleModel]
    try:
        parsed_obj = partial_output_cls.model_validate_json(content)
    except (ValidationError, ValueError):
        try:
            repaired_json = _repair_incomplete_json(content)
            parsed_obj = partial_output_cls.model_validate_json(repaired_json)
        except (ValidationError, ValueError):
            parsed_obj = _extract_partial_list_progress(
                content, output_cls, cur_object, partial_output_cls
            )

    # If we still couldn't parse anything, use previous object
    if parsed_obj is None:
        if cur_object is not None:
            return cur_object
        return partial_output_cls()

    # Use incremental comparison that considers list progress
    try:
        return output_cls.model_validate(parsed_obj.model_dump(exclude_unset=True))
    except ValidationError:
        return parsed_obj


def _extract_list_fields(content: str) -> List[Tuple[str, str]]:
    """
    Extract top-level ``"field": [...]`` slices from potentially incomplete JSON.

    Respects JSON string quoting, escapes, and nested brackets so ``]`` inside
    string values or nested arrays does not truncate the outer list content.
    """
    results: List[Tuple[str, str]] = []
    n = len(content)
    i = 0
    brace_depth = 0
    bracket_depth = 0

    while i < n:
        ch = content[i]
        if ch == '"':
            # Scan quoted JSON string
            j = i + 1
            escape = False
            while j < n:
                cj = content[j]
                if escape:
                    escape = False
                elif cj == "\\":
                    escape = True
                elif cj == '"':
                    break
                j += 1
            if j >= n:
                break

            key = content[i + 1 : j]
            k = j + 1
            while k < n and content[k].isspace():
                k += 1

            if brace_depth <= 1 and bracket_depth == 0 and k < n and content[k] == ":":
                k += 1
                while k < n and content[k].isspace():
                    k += 1
                if k < n and content[k] == "[":
                    # Capture until matching ']' or end of incomplete stream
                    m = k + 1
                    arr_depth = 1
                    arr_in_str = False
                    arr_escape = False
                    while m < n and arr_depth > 0:
                        cm = content[m]
                        if arr_in_str:
                            if arr_escape:
                                arr_escape = False
                            elif cm == "\\":
                                arr_escape = True
                            elif cm == '"':
                                arr_in_str = False
                        else:
                            if cm == '"':
                                arr_in_str = True
                            elif cm == "[":
                                arr_depth += 1
                            elif cm == "]":
                                arr_depth -= 1
                        m += 1
                    list_slice = (
                        content[k + 1 : m - 1] if arr_depth == 0 else content[k + 1 :]
                    )
                    results.append((key, list_slice))
                    i = m
                    continue

            i = j + 1
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        i += 1

    return results


def _extract_top_level_object_slices(list_content: str) -> List[str]:
    """
    Extract top-level ``{...}`` object slices (including a trailing incomplete one)
    from partial list content while respecting nested braces and string literals.
    """
    slices: List[str] = []
    in_string = False
    escape = False
    depth = 0
    obj_start: Optional[int] = None

    for i, ch in enumerate(list_content):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    slices.append(list_content[obj_start : i + 1])
                    obj_start = None

    if depth > 0 and obj_start is not None:
        slices.append(list_content[obj_start:])

    return slices


def _extract_partial_list_progress(
    content: str,
    output_cls: Type[Model],
    cur_object: Optional[Union[Model, FlexibleModel]],
    partial_output_cls: Type[FlexibleModel],
) -> Optional[FlexibleModel]:
    """
    Try to extract partial list progress from incomplete JSON.

    This attempts to build upon the current object by detecting partial
    list additions even when JSON is malformed.
    """
    if not isinstance(content, str) or cur_object is None:
        return None

    try:
        matches = _extract_list_fields(content)

        if not matches:
            return None

        # Start with current object data
        if isinstance(cur_object, FlexibleModel):
            current_data = cur_object.model_dump(exclude_unset=True)
        elif hasattr(cur_object, "model_dump"):
            current_data = cur_object.model_dump()
        else:
            current_data = {}

        updated = False
        for field_name, list_content in matches:
            if (
                hasattr(output_cls, "model_fields")
                and field_name in output_cls.model_fields
            ):
                # Try to parse individual items from the list content
                items = _parse_partial_list_items(list_content, field_name, output_cls)
                if items:
                    current_data[field_name] = items
                    updated = True

        if not updated:
            return None

        # Try to create object with updated data
        return partial_output_cls.model_validate(current_data)

    except Exception:
        return None


def _parse_partial_list_items(
    list_content: str, field_name: str, output_cls: Type[Model]
) -> List[Any]:
    """Parse individual items from partial list content."""
    try:
        items: List[Any] = []
        object_matches = _extract_top_level_object_slices(list_content)

        for obj_str in object_matches:
            try:
                # Try to parse as complete JSON object
                obj_data = json.loads(obj_str)
                if isinstance(obj_data, dict):
                    items.append(obj_data)
            except (json.JSONDecodeError, SyntaxError):
                # Try to repair and parse
                try:
                    repaired = _repair_incomplete_json(obj_str)
                    obj_data = json.loads(repaired)
                    if isinstance(obj_data, dict):
                        items.append(obj_data)
                except (json.JSONDecodeError, SyntaxError):
                    continue

        return items

    except Exception:
        return []
