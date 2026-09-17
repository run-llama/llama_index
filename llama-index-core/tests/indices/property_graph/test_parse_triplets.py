from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn


def test_parse_triplets_none_returns_empty() -> None:
    assert default_parse_triplets_fn(None) == []  # type: ignore[arg-type]


def test_parse_triplets_empty_returns_empty() -> None:
    assert default_parse_triplets_fn("") == []
    assert default_parse_triplets_fn("   ") == []


def test_parse_triplets_valid_line() -> None:
    triplets = default_parse_triplets_fn('(Alice, knows, Bob)')
    assert triplets == [("Alice", "Knows", "Bob")]
