"""General node utils."""


import logging
from typing import List, Optional

from llama_index.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    MetadataMode,
    NodeRelationship,
    TextNode,
)
from llama_index.text_splitter.types import MetadataAwareTextSplitter, SplitterType
from llama_index.utils import truncate_text

logger = logging.getLogger(__name__)


def build_nodes_from_splits(
    text_splits: List[str],
    document: BaseNode,
    include_metadata: bool = True,
    include_prev_next_rel: bool = False,
    ref_doc: Optional[BaseNode] = None,
) -> List[TextNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document

    nodes: List[TextNode] = []
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        node_metadata = {}
        if include_metadata:
            node_metadata = document.metadata

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                text=text_chunk,
                embedding=document.embedding,
                metadata=node_metadata,
                image=document.image,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document):
            node = TextNode(
                text=text_chunk,
                embedding=document.embedding,
                metadata=node_metadata,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(node)
        elif isinstance(document, TextNode):
            node = TextNode(
                text=text_chunk,
                embedding=document.embedding,
                metadata=node_metadata,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
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


def get_nodes_from_document(
    document: BaseNode,
    text_splitter: SplitterType,
    include_metadata: bool = True,
    include_prev_next_rel: bool = False,
) -> List[TextNode]:
    """Get nodes from document.

    NOTE: this function has been deprecated, please use
    get_nodes_from_node which supports both documents/nodes.

    """
    return get_nodes_from_node(
        document,
        text_splitter,
        include_metadata=include_metadata,
        include_prev_next_rel=include_prev_next_rel,
        ref_doc=document,
    )


def get_nodes_from_node(
    node: BaseNode,
    text_splitter: SplitterType,
    include_metadata: bool = True,
    include_prev_next_rel: bool = False,
    ref_doc: Optional[BaseNode] = None,
) -> List[TextNode]:
    """Get nodes from document."""
    if include_metadata:
        if isinstance(text_splitter, MetadataAwareTextSplitter):
            embed_metadata_str = node.get_metadata_str(mode=MetadataMode.EMBED)
            llm_metadata_str = node.get_metadata_str(mode=MetadataMode.LLM)

            # use the longest metadata str for splitting
            if len(embed_metadata_str) > len(llm_metadata_str):
                metadata_str = embed_metadata_str
            else:
                metadata_str = llm_metadata_str

            text_splits = text_splitter.split_text_metadata_aware(
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                metadata_str=metadata_str,
            )
        else:
            logger.warning(
                f"include_metadata is set to True but {text_splitter} "
                "is not metadata-aware."
                "Node content length may exceed expected chunk size."
                "Try lowering the chunk size or using a metadata-aware text splitter "
                "if this is a problem."
            )

            text_splits = text_splitter.split_text(
                node.get_content(metadata_mode=MetadataMode.NONE),
            )
    else:
        text_splits = text_splitter.split_text(
            node.get_content(metadata_mode=MetadataMode.NONE),
        )

    return build_nodes_from_splits(
        text_splits,
        node,
        include_metadata=include_metadata,
        include_prev_next_rel=include_prev_next_rel,
        ref_doc=ref_doc,
    )
