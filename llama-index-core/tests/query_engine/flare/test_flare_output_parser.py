"""Tests for the FLARE query task output parser."""

from llama_index.core.query_engine.flare.output_parser import QueryTaskOutputParser


def test_parse_search_tags() -> None:
    """A well-formed [Search(query)] tag still produces a query task."""
    parser = QueryTaskOutputParser()
    output = "Search for [Search(Where is reception?)] now"

    tasks = parser.parse(output)

    assert len(tasks) == 1
    assert tasks[0].query_str == "Where is reception?"
    assert (
        output[tasks[0].start_idx : tasks[0].end_idx + 1]
        == "[Search(Where is reception?)]"
    )


def test_ignore_bracketed_text_without_a_query() -> None:
    """Ordinary bracketed text is not a search tag and must not crash parsing."""
    parser = QueryTaskOutputParser()
    outputs = (
        "Your train arrives at [Station Name] at 14:30.",
        "See [1] and [docs](https://example.com) for details.",
        "A stray ] before any opening bracket",
        "An unclosed [bracket without a query",
        "A query with a missing [Search(Where is reception?] parenthesis",
    )

    for output in outputs:
        assert parser.parse(output) == []


def test_parse_only_well_formed_search_tags() -> None:
    """Bracketed prose around a real tag is ignored, not parsed as a query."""
    parser = QueryTaskOutputParser()
    output = "[Station Name] then [Search(Where is reception?)] and [1]"

    tasks = parser.parse(output)

    assert len(tasks) == 1
    assert tasks[0].query_str == "Where is reception?"
