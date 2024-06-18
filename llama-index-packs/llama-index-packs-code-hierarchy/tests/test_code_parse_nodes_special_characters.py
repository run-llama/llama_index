"""Test CodeHierarchyNodeParser with a special character on the code."""

from typing import List

from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.core.schema import TextNode


def test_special_character() -> None:
    """Test case for code splitting using python and add a special character in the code."""
    code_splitter = CodeHierarchyNodeParser(
        language="python", skeleton=False, chunk_min_characters=0
    )

    # example with special character code ¥
    text = """\
def print_special_character():
    print("Particulate Matter 10 (¥g/m3)")


def function_that_was_cut():
    print("This function was cut from the original file")"""

    text_node = TextNode(
        text=text,
        metadata={
            "module": "example.foo",
        },
    )

    nodes: List[TextNode] = code_splitter.get_nodes_from_documents([text_node])

    assert len(nodes) == 3
    # The last node is didn't have the first character, because the length of utf-8 special character ¥ is 2
    assert (
        nodes[2].text
        == """\
def function_that_was_cut():
    print("This function was cut from the original file")"""
    )
    assert nodes[2].metadata["start_byte"] == 77
    assert nodes[2].metadata["end_byte"] == 163
