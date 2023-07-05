"""General node utils."""


import logging
from typing import List

from llama_index.langchain_helpers.text_splitter import (
    TextSplit,
    TextSplitter,
    TokenTextSplitter,
)
from llama_index.schema import Document, ImageDocument
from llama_index.schema import (
    BaseNode,
    NodeRelationship,
    ImageNode,
    TextNode,
    MetadataMode,
)
from llama_index.utils import truncate_text

logger = logging.getLogger(__name__)


def get_text_splits_from_document(
    document: BaseNode,
    text_splitter: TextSplitter,
    include_metadata: bool = True,
) -> List[TextSplit]:
    """Break the document into chunks with additional info."""
    # TODO: clean up since this only exists due to the diff w LangChain's TextSplitter
    if isinstance(text_splitter, TokenTextSplitter):
        # use this to extract extra information about the chunks
        text_splits = text_splitter.split_text_with_overlaps(
            document.get_content(metadata_mode=MetadataMode.NONE),
            metadata_str=document.get_metadata_str(mode=MetadataMode.LLM)
            if include_metadata
            else None,
        )
    else:
        text_chunks = text_splitter.split_text(
            document.get_content(metadata_mode=MetadataMode.NONE),
        )
        text_splits = [TextSplit(text_chunk=text_chunk) for text_chunk in text_chunks]

    return text_splits


def get_nodes_from_document(
    document: BaseNode,
    text_splitter: TextSplitter,
    include_metadata: bool = True,
    include_prev_next_rel: bool = False,
) -> List[TextNode]:
    """Get nodes from document."""
    text_splits = get_text_splits_from_document(
        document=document,
        text_splitter=text_splitter,
        include_metadata=include_metadata,
    )

    nodes: List[TextNode] = []
    index_counter = 0
    for i, text_split in enumerate(text_splits):
        text_chunk = text_split.text_chunk
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")
        start_char_idx = None
        end_char_idx = None
        if text_split.num_char_overlap is not None:
            start_char_idx = index_counter - text_split.num_char_overlap
            end_char_idx = index_counter - text_split.num_char_overlap + len(text_chunk)
        index_counter += len(text_chunk) + 1

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                text=text_chunk,
                embedding=document.embedding,
                metadata=document.metadata if include_metadata else {},
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                image=document.image,
                relationships={
                    NodeRelationship.SOURCE: document.as_related_node_info()
                },
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document):
            node = TextNode(
                text=text_chunk,
                embedding=document.embedding,
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                metadata=document.metadata if include_metadata else {},
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                text_template=document.text_template,
                relationships={
                    NodeRelationship.SOURCE: document.as_related_node_info()
                },
            )
            nodes.append(node)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")

    # if include_prev_next_rel, then add prev/next relationships
    if include_prev_next_rel:
        for i, node in enumerate(nodes):
            if i > 0:
                node.relationships[NodeRelationship.PREVIOUS] = nodes[
                    i - 1
                ].as_related_node_info()
            if i < len(nodes) - 1:
                node.relationships[NodeRelationship.NEXT] = nodes[
                    i + 1
                ].as_related_node_info()

    return nodes
