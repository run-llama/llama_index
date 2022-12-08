"""Test indices/utils.py."""
from gpt_index.indices.data_structs import Node
from gpt_index.indices.utils import expand_tokens_with_subtokens, get_text_from_nodes


def test_get_text_from_nodes() -> None:
    """Get text from nodes. Used by tree-structured indices."""
    node1 = Node(text="This is a test. Hello my name is John.")
    node2 = Node(text="This is another test. Hello world!")
    node_list = [node1, node2]
    response = get_text_from_nodes(node_list)
    assert response == (
        "This is a test. Hello my name is John.\n" "This is another test. Hello world!"
    )


def test_expand_tokens_with_subtokens() -> None:
    """Test expand tokens."""
    tokens = {"foo bar", "baz", "hello hello world bye"}
    keywords = expand_tokens_with_subtokens(tokens)
    assert keywords == {
        "foo bar",
        "foo",
        "bar",
        "baz",
        "hello hello world bye",
        "hello",
        "world",
        "bye",
    }
