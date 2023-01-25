"""General node utils."""


from typing import List

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import BaseDocument


def get_nodes_from_document(
    document: BaseDocument,
    text_splitter: TokenTextSplitter,
    start_idx: int = 0,
    include_extra_info: bool = True,
) -> List[Node]:
    """Add document to index."""
    text_chunks = text_splitter.split_text(
        document.get_text(),
        extra_info_str=document.extra_info_str if include_extra_info else None,
    )
    nodes = []
    index_counter = 0
    for i, text_chunk in enumerate(text_chunks):
        fmt_text_chunk = truncate_text(text_chunk, 50)
        print(f"> Adding chunk: {fmt_text_chunk}")
        index_pos_info = {
            "start": index_counter,  # NOTE: start is inclusive
            "end": index_counter + len(text_chunk),  # NOTE: end is exclusive
        }
        index_counter += len(text_chunk) + 1
        # if embedding specified in document, pass it to the Node
        node = Node(
            text=text_chunk,
            index=start_idx + i,
            ref_doc_id=document.get_doc_id(),
            embedding=document.embedding,
            extra_info=document.extra_info if include_extra_info else None,
            node_info=index_pos_info,
        )
        nodes.append(node)
    return nodes
