"""Test node."""

from gpt_index.data_structs.node_v2 import Node


def test_get_text_from_node() -> None:
    """Test get text from node."""
    extra_info = {"foo": "bar", "test": "test2", "test3": 2}
    # test normal node
    node = Node("hello world", extra_info=extra_info)
    text = node.get_text()
    assert text == ("foo: bar\n" "test: test2\n" "test3: 2\n\n" "hello world")

    # Test that text is returned when there is no extra info
    node = Node(
        "hello world", extra_info=extra_info, exclude_extra_info_keys_from_text=["foo"]
    )
    text = node.get_text()
    assert text == ("test: test2\n" "test3: 2\n\n" "hello world")

    # Test that text is returned when there is extra info
    node = Node("testing", extra_info=extra_info, exclude_all_extra_info_from_text=True)
    text = node.get_text()
    assert text == "testing"
