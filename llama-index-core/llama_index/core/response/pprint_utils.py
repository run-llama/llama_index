"""Utils for pretty print."""

import textwrap
from pprint import pprint
from typing import Any, Dict

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from llama_index.core.utils import truncate_text


def pprint_metadata(metadata: Dict[str, Any]) -> None:
    """Display metadata for jupyter notebook."""
    pprint(metadata)


def pprint_source_node(
    source_node: NodeWithScore, source_length: int = 350, wrap_width: int = 70
) -> None:
    """Display source node for jupyter notebook."""
    source_text_fmt = truncate_text(
        source_node.node.get_content().strip(), source_length
    )
    print(f"Node ID: {source_node.node.node_id}")
    print(f"Similarity: {source_node.score}")
    print(textwrap.fill(f"Text: {source_text_fmt}\n", width=wrap_width))


def pprint_response(
    response: Response,
    source_length: int = 350,
    wrap_width: int = 70,
    show_source: bool = False,
) -> None:
    """Pretty print response for jupyter notebook."""
    if response.response is None:
        response_text = "None"
    else:
        response_text = response.response.strip()

    response_text = f"Final Response: {response_text}"
    print(textwrap.fill(response_text, width=wrap_width))

    if show_source:
        for ind, source_node in enumerate(response.source_nodes):
            print("_" * wrap_width)
            print(f"Source Node {ind + 1}/{len(response.source_nodes)}")
            pprint_source_node(
                source_node, source_length=source_length, wrap_width=wrap_width
            )
