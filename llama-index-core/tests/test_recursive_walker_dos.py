"""
Regression tests for the unguarded recursive walkers in `llama-index-core`.

These mirror the defect class fixed by PR #18877 for `JSONReader` (CVE-2025-5302 /
CVE-2025-5472) and cover the remaining sinks reported in the huntr advisory at
https://huntr.com/bounties/412a8af2-d173-402d-aa9b-f4989bfaa4c7:

1. ChatMessage._recursive_serialization (via serialize_additional_kwargs)
2. SelectionOutputParser._filter_dict (via parse)
3. graph_stores.utils.value_sanitize
4. TextToCypherRetriever._clean_query_output
"""

import json
import sys

import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.graph_stores.utils import value_sanitize
from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.output_parsers.base import OutputParserException
from llama_index.core.output_parsers.selection import SelectionOutputParser


def _make_nested(depth: int):
    d = "leaf"
    for _ in range(depth):
        d = {"k": d}
    return d


# Use a depth comfortably above Python's default recursion limit (1000) so the
# test deterministically exercises the RecursionError guard regardless of any
# constant per-call frame overhead inside the walkers.
_DEEP = sys.getrecursionlimit() + 500


def test_chat_message_additional_kwargs_deeply_nested_does_not_raise() -> None:
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="ok",
        additional_kwargs={"x": _make_nested(_DEEP)},
    )

    # Before the fix, this raised PydanticSerializationError wrapping
    # RecursionError, which broke every downstream persist / event-dispatch.
    dumped = msg.model_dump()

    assert dumped["additional_kwargs"] == {}


def test_selection_output_parser_deeply_nested_raises_output_parser_exception() -> None:
    parser = SelectionOutputParser()
    # JSON missing both `choice` and `reason` falls into the recursive
    # `_filter_dict` path.
    poison = json.dumps(_make_nested(_DEEP))

    with pytest.raises(OutputParserException, match="deeply nested"):
        parser.parse(poison)


def test_value_sanitize_deeply_nested_returns_none() -> None:
    # Before the fix this raised a raw RecursionError out of every graph
    # property-cleaning call.
    assert value_sanitize(_make_nested(_DEEP)) is None


def test_text_to_cypher_clean_query_output_deeply_nested_returns_none() -> None:
    class _Stub:
        allowed_output_fields = ["only_a_safe_key"]
        _clean_query_output = TextToCypherRetriever._clean_query_output
        _clean_query_output_impl = TextToCypherRetriever._clean_query_output_impl

    # Before the fix this raised a raw RecursionError out of retrieve() /
    # aretrieve() on every poisoned input.
    assert _Stub()._clean_query_output(_make_nested(_DEEP)) is None
