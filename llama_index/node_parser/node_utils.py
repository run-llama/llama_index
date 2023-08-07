"""General node utils."""


import logging
from typing import List

from llama_index.schema import (BaseNode, Document, ImageDocument, ImageNode,
                                MetadataMode, NodeRelationship, TextNode)
from llama_index.text_splitter import TextSplitter
from llama_index.utils import truncate_text

logger = logging.getLogger(__name__)


def get_nodes_from_document(
    document: BaseNode,
    text_splitter: TextSplitter,
    include_metadata: bool = True,
    include_prev_next_rel: bool = False,
) -> List[TextNode]:
    """Get nodes from document."""
    text_splits = text_splitter.split_text(
        document.get_content(metadata_mode=MetadataMode.NONE),
    )

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
                relationships={
                    NodeRelationship.SOURCE: document.as_related_node_info()
                },
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
