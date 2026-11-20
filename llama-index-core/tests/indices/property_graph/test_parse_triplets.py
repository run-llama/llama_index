"""Tests for triplet parsing utilities."""

from llama_index.core.indices.property_graph.utils import (
    _extract_triplet_candidates,
    default_parse_triplets_fn,
)
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex


class TestExtractTripletCandidates:
    def test_single_triplet(self):
        result = _extract_triplet_candidates("(Alice, knows, Bob)")
        assert result == ["Alice, knows, Bob"]

    def test_multiple_triplets(self):
        result = _extract_triplet_candidates(
            "(Alice, knows, Bob) (Bob, knows, Carol)"
        )
        assert result == ["Alice, knows, Bob", "Bob, knows, Carol"]

    def test_double_parentheses(self):
        result = _extract_triplet_candidates("((Alice, knows, Bob))")
        assert result == ["Alice, knows, Bob"]

    def test_no_parens(self):
        result = _extract_triplet_candidates("no triplets here")
        assert result == []

    def test_nested_parens_extracts_innermost(self):
        result = _extract_triplet_candidates("(f(x), equals, y)")
        assert result == ["x"]

    def test_empty_parens(self):
        result = _extract_triplet_candidates("()")
        assert result == [""]


class TestDefaultParseTripletsFn:
    def test_basic_triplet(self):
        result = default_parse_triplets_fn("(Alice, works at, Google)")
        assert result == [("Alice", "Works at", "Google")]

    def test_multiple_triplets_one_line(self):
        result = default_parse_triplets_fn(
            "(Alice, knows, Bob) (Bob, knows, Carol)"
        )
        assert result == [("Alice", "Knows", "Bob"), ("Bob", "Knows", "Carol")]

    def test_double_parentheses(self):
        result = default_parse_triplets_fn("((Alice, knows, Bob))")
        assert result == [("Alice", "Knows", "Bob")]

    def test_multiline(self):
        text = "(Alice, knows, Bob)\n(Paris, capital of, France)"
        result = default_parse_triplets_fn(text)
        assert result == [
            ("Alice", "Knows", "Bob"),
            ("Paris", "Capital of", "France"),
        ]

    def test_skips_junk_lines(self):
        text = "some text\n\n(Alice, knows, Bob)\nmore junk"
        result = default_parse_triplets_fn(text)
        assert result == [("Alice", "Knows", "Bob")]

    def test_skips_non_three_token(self):
        result = default_parse_triplets_fn("(Alice, knows)")
        assert result == []

    def test_skips_too_long_tokens(self):
        long = "x" * 200
        result = default_parse_triplets_fn(f"({long}, knows, Bob)")
        assert result == []

    def test_skips_empty_tokens(self):
        result = default_parse_triplets_fn("(, knows, Bob)")
        assert result == []

    def test_strips_quotes(self):
        result = default_parse_triplets_fn('("Alice", "knows", "Bob")')
        assert result == [("Alice", "Knows", "Bob")]

    def test_numbered_list(self):
        text = "1. (Alice, knows, Bob)\n2. (Bob, knows, Carol)"
        result = default_parse_triplets_fn(text)
        assert result == [
            ("Alice", "Knows", "Bob"),
            ("Bob", "Knows", "Carol"),
        ]


class TestKnowledgeGraphParseTripletResponse:
    """Verify the sibling parser in KnowledgeGraphIndex uses the same fix."""

    def test_multiple_triplets_one_line(self):
        result = KnowledgeGraphIndex._parse_triplet_response(
            "(Alice, knows, Bob) (Bob, knows, Carol)"
        )
        assert result == [("Alice", "Knows", "Bob"), ("Bob", "Knows", "Carol")]

    def test_double_parentheses(self):
        result = KnowledgeGraphIndex._parse_triplet_response(
            "((Alice, knows, Bob))"
        )
        assert result == [("Alice", "Knows", "Bob")]

    def test_basic_still_works(self):
        result = KnowledgeGraphIndex._parse_triplet_response(
            "(Alice, works at, Google)"
        )
        assert result == [("Alice", "Works at", "Google")]
