"""
Simplified streaming utilities for processing structured outputs from message content.

This module provides utilities for processing streaming responses that contain
structured data directly in the message content (not in function calls).
"""

from typing import Optional, Type, Union

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
    try:
        parsed_obj = partial_output_cls.model_validate_json(content)
    except (ValidationError, ValueError):
        try:
            repaired_json = _repair_incomplete_json(content)
            parsed_obj = partial_output_cls.model_validate_json(repaired_json)
        except (ValidationError, ValueError):
            extracted_obj = _extract_partial_list_progress(
                content, output_cls, cur_object, partial_output_cls
            )
            parsed_obj = (
                extracted_obj if extracted_obj is not None else partial_output_cls()
            )

    # If we still couldn't parse anything, use previous object
    if parsed_obj is None:
        if cur_object is not None:
            return cur_object
        else:
            return partial_output_cls()

    # Use incremental comparison that considers list progress
    try:
        return output_cls.model_validate(parsed_obj.model_dump(exclude_unset=True))
    except ValidationError:
        return parsed_obj


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
        import re

        # Try to extract list patterns from incomplete JSON
        # Look for patterns like: "jokes": [{"setup": "...", "punchline": "..."}
        list_pattern = r'"(\w+)":\s*\[([^\]]*)'
        matches = re.findall(list_pattern, content)

        if not matches:
            return None

        # Start with current object data
        current_data = (
            cur_object.model_dump() if hasattr(cur_object, "model_dump") else {}
        )

        for field_name, list_content in matches:
            if (
                hasattr(output_cls, "model_fields")
                and field_name in output_cls.model_fields
            ):
                # Try to parse individual items from the list content
                items = _parse_partial_list_items(list_content, field_name, output_cls)
                if items:
                    current_data[field_name] = items

        # Try to create object with updated data
        return partial_output_cls.model_validate(current_data)

    except Exception:
        return None


def _parse_partial_list_items(
    list_content: str, field_name: str, output_cls: Type[Model]
) -> list:
    """
    Parse individual items from partial list content.
    """
    try:
        import json
        import re

        items = []

        # Look for complete object patterns within the list
        # Pattern: {"key": "value", "key2": "value2"}
        object_pattern = r"\{[^{}]*\}"
        object_matches = re.findall(object_pattern, list_content)

        for obj_str in object_matches:
            try:
                # Try to parse as complete JSON object
                obj_data = json.loads(obj_str)
                items.append(obj_data)
            except (json.JSONDecodeError, SyntaxError):
                # Try to repair and parse
                try:
                    repaired = _repair_incomplete_json(obj_str)
                    obj_data = json.loads(repaired)
                    items.append(obj_data)
                except (json.JSONDecodeError, SyntaxError):
                    continue

        return items

    except Exception:
        return []
