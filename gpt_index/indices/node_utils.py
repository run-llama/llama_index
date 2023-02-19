"""General node utils."""


import logging
from typing import List

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.schema import BaseDocument


def get_nodes_from_document(
    document: BaseDocument,
    text_splitter: TextSplitter,
    start_idx: int = 0,
    include_extra_info: bool = True,
) -> List[Node]:
    """Add document to index."""
    text_chunks = []
    text_chunks_with_overlap = []
    if hasattr(text_splitter, "split_text_with_overlaps"):
        # use this to extract extra information about the chunks
        text_chunks_with_overlap = text_splitter.split_text_with_overlaps(
            document.get_text(),
            extra_info_str=document.extra_info_str if include_extra_info else None,
        )
        text_chunks = [text_split.text_chunk for text_split in text_chunks_with_overlap]
    else:
        text_chunks = text_splitter.split_text(
            document.get_text(),
            extra_info_str=document.extra_info_str if include_extra_info else None,
        )

    nodes = []
    index_counter = 0
    for i, text_chunk in enumerate(text_chunks):
        logging.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")
        index_pos_info = None
        if len(text_chunks_with_overlap) > 0:
            text_split = text_chunks_with_overlap[i]
            index_pos_info = {
                # NOTE: start is inclusive, end is exclusive
                "start": index_counter - text_split.num_char_overlap,
                "end": index_counter - text_split.num_char_overlap + len(text_chunk),
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
