from typing import List, Optional

from llama_index.schema import BaseNode
from llama_index.text_splitter import TokenTextSplitter
from llama_index.text_splitter.utils import truncate_text


def get_numbered_text_from_nodes(
    node_list: List[BaseNode],
    text_splitter: Optional[TokenTextSplitter] = None,
) -> str:
    """Get text from nodes in the format of a numbered list.

    Used by tree-structured indices.

    """
    results = []
    number = 1
    for node in node_list:
        node_text = " ".join(node.get_content().splitlines())
        if text_splitter is not None:
            node_text = truncate_text(node_text, text_splitter)
        text = f"({number}) {node_text}"
        results.append(text)
        number += 1
    return "\n\n".join(results)
