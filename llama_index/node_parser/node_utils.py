"""General node utils."""


import logging
from typing import List, Optional

from llama_index.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    NodeRelationship,
    TextNode,
)
from llama_index.utils import truncate_text

logger = logging.getLogger(__name__)


def build_nodes_from_splits(
    text_splits: List[str],
    document: BaseNode,
    ref_doc: Optional[BaseNode] = None,
) -> List[TextNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document

    nodes: List[TextNode] = []
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                text=text_chunk,
                embedding=document.embedding,
                image=document.image,
                image_path=document.image_path,
                image_url=document.image_url,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document):
            node = TextNode(
                text=text_chunk,
                embedding=document.embedding,
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

    return nodes


def build_nodes_from_prechunked(
        document: BaseNode, ## 包含chunk数据的具体文档
        ref_doc: Optional[BaseNode] = None,  ## 文档抽象 源
) -> List[TextNode]:
    """Build nodes from pre chunked doc."""
    """Real Parser Service is independed from llama-index proj."""
    ref_doc = ref_doc or document

    nodes: List[TextNode] = []

    if isinstance(document, ImageDocument): #TODO
        image_node = ImageNode(
            text=text_chunk,
            embedding=document.embedding,
            image=document.image,
            image_path=document.image_path,
            image_url=document.image_url,
            excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
            metadata_seperator=document.metadata_seperator,
            metadata_template=document.metadata_template,
            text_template=document.text_template,
            relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
        )
        nodes.append(image_node)
    elif isinstance(document, Document):
        node = TextNode(
            text=text_chunk,
            embedding=document.embedding,
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

    return nodes

def docExist(document: BaseNode) -> bool:
    #TODO  may be better in other position
    return False

