"""General node utils."""


import logging
from typing import List

from llama_index.data_structs.node import DocumentRelationship, ImageNode, Node
from llama_index.langchain_helpers.text_splitter import (
    TextSplit,
    TextSplitter,
    TokenTextSplitter,
)
from llama_index.readers.schema.base import ImageDocument
from llama_index.schema import BaseDocument
from llama_index.utils import truncate_text

logger = logging.getLogger(__name__)


def get_text_splits_from_document(
    document: BaseDocument,
    text_splitter: TextSplitter,
    include_extra_info: bool = True,
) -> List[TextSplit]:
    """Break the document into chunks with additional info."""
    # TODO: clean up since this only exists due to the diff w LangChain's TextSplitter
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
    include_extra_info: bool = True,
    include_prev_next_rel: bool = False,
) -> List[Node]:
    """Get nodes from document."""
    text_splits = get_text_splits_from_document(
        document=document,
        text_splitter=text_splitter,
        include_extra_info=include_extra_info,
    )

    nodes: List[Node] = []
    index_counter = 0
    for i, text_split in enumerate(text_splits):
        text_chunk = text_split.text_chunk
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")
        index_pos_info = None
        if text_split.num_char_overlap is not None:
            index_pos_info = {
                # NOTE: start is inclusive, end is exclusive
                "start": index_counter - text_split.num_char_overlap,
                "end": index_counter - text_split.num_char_overlap + len(text_chunk),
            }
        index_counter += len(text_chunk) + 1

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                text=text_chunk,
                embedding=document.embedding,
                extra_info=document.extra_info if include_extra_info else None,
                node_info=index_pos_info,
                image=document.image,
                relationships={DocumentRelationship.SOURCE: document.get_doc_id()},
            )
            nodes.append(image_node)  # type: ignore
        else:
            node = Node(
                text=text_chunk,
                embedding=document.embedding,
                extra_info=document.extra_info if include_extra_info else None,
                node_info=index_pos_info,
                relationships={DocumentRelationship.SOURCE: document.get_doc_id()},
            )
            nodes.append(node)

    # if include_prev_next_rel, then add prev/next relationships
    if include_prev_next_rel:
        for i, node in enumerate(nodes):
            if i > 0:
                node.relationships[DocumentRelationship.PREVIOUS] = nodes[
                    i - 1
                ].get_doc_id()
            if i < len(nodes) - 1:
                node.relationships[DocumentRelationship.NEXT] = nodes[
                    i + 1
                ].get_doc_id()

    return nodes
