"""Requires llama-index-core + distil-llm installed. Run: `pytest` in the package root."""

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode

from distil.mcp_server import load_restore
from llama_index.postprocessor.distil import DistilNodePostprocessor


def _long_node() -> NodeWithScore:
    text = "\n".join([f"log line {i}: some detail" for i in range(40)])
    return NodeWithScore(node=TextNode(text=text), score=1.0)


def test_compresses_and_is_reversible():
    nws = _long_node()
    original = nws.node.text
    out = DistilNodePostprocessor()._postprocess_nodes([nws], QueryBundle(query_str="detail"))
    compressed = out[0].node.text
    assert compressed != original
    assert "handle=" in compressed  # dropped middle replaced by a recoverable marker
    import re

    handle = re.search(r"handle=([0-9a-f]{8})", compressed).group(1)
    assert load_restore(handle) == original  # exact bytes recoverable


def test_short_node_untouched():
    nws = NodeWithScore(node=TextNode(text="one\ntwo"), score=1.0)
    out = DistilNodePostprocessor()._postprocess_nodes([nws], None)
    assert out[0].node.text == "one\ntwo"
