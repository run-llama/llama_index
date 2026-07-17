from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode

from llama_index.postprocessor.distil import DistilNodePostprocessor


def test_class_is_a_node_postprocessor():
    names = [b.__name__ for b in DistilNodePostprocessor.__mro__]
    assert BaseNodePostprocessor.__name__ in names


def test_long_node_is_compressed_and_recoverable():
    text = "\n".join(f"line {i}: routine tool output with no decision content" for i in range(30))
    node = NodeWithScore(node=TextNode(text=text))
    out = DistilNodePostprocessor()._postprocess_nodes([node])
    assert "handle=" in out[0].node.text  # middle folded behind a recoverable handle
    assert len(out[0].node.text) < len(text)


def test_short_node_passes_through_unchanged():
    node = NodeWithScore(node=TextNode(text="one short line"))
    out = DistilNodePostprocessor()._postprocess_nodes([node])
    assert out[0].node.text == "one short line"
