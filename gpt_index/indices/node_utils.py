"""General node utils."""


import logging
from typing import List

from gpt_index.data_structs.data_structs import Node
from gpt_index.langchain_helpers.text_splitter import (
    TextSplit,
    TextSplitter,
    TokenTextSplitter,
)
from gpt_index.schema import BaseDocument
from gpt_index.utils import truncate_text


def get_text_splits_from_document(
    document: BaseDocument,
    text_splitter: TextSplitter,
    include_extra_info: bool = True,
) -> List[TextSplit]:
    """Break the document into chunks with additional info."""
    # TODO: clean up since this only exists due to the diff w LangChain's TextSplitter
    text_splits = []
    if isinstance(text_splitter, TokenTextSplitter):
        # use this to extract extra information about the chunks
        text_splits = text_splitter.split_text_with_overlaps(
            document.get_text(),
            extra_info_str=document.extra_info_str if include_extra_info else None,
        )
    else:
        text_chunks = text_splitter.split_text(
            document.get_text(),
        )
        text_splits = [TextSplit(text_chunk=text_chunk) for text_chunk in text_chunks]

    return text_splits


def get_nodes_from_document(
    document: BaseDocument,
    text_splitter: TextSplitter,
    start_idx: int = 0,
    include_extra_info: bool = True,
) -> List[Node]:
    """Add document to index."""
    text_splits = get_text_splits_from_document(
        document=document,
        text_splitter=text_splitter,
        include_extra_info=include_extra_info,
    )

    nodes = []
    index_counter = 0
    for i, text_split in enumerate(text_splits):
        text_chunk = text_split.text_chunk
        logging.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")
        index_pos_info = None
        if text_split.num_char_overlap is not None:
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
