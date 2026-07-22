"""Tests for the FLARE output parsers' not-implemented contract."""

import pytest

from llama_index.core.query_engine.flare.output_parser import (
    IsDoneOutputParser,
    QueryTaskOutputParser,
)


def test_is_done_output_parser_format_raises_descriptive() -> None:
    """IsDoneOutputParser.format must raise NotImplementedError with a message."""
    parser = IsDoneOutputParser()
    with pytest.raises(NotImplementedError, match="format"):
        parser.format("query")


def test_query_task_output_parser_format_raises_descriptive() -> None:
    """QueryTaskOutputParser.format must raise NotImplementedError with a message."""
    parser = QueryTaskOutputParser()
    with pytest.raises(NotImplementedError, match="format"):
        parser.format("query")
