"""General node utils."""

import logging
import uuid
from typing import List, Optional, Protocol, runtime_checkable

from llama_index.legacy.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    NodeRelationship,
    TextNode,
)
from llama_index.legacy.utils import truncate_text

logger = logging.getLogger(__name__)


@runtime_checkable
class IdFuncCallable(Protocol):
    def __call__(self, i: int, doc: BaseNode) -> str:
        ...


def default_id_func(i: int, doc: BaseNode) -> str:
    return str(uuid.uuid4())


def build_nodes_from_splits(
    text_splits: List[str],
    document: BaseNode,
    ref_doc: Optional[BaseNode] = None,
    id_func: Optional[IdFuncCallable] = None,
) -> List[TextNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document
    id_func = id_func or default_id_func
    nodes: List[TextNode] = []
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                id_=id_func(i, document),
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
                id_=id_func(i, document),
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
                id_=id_func(i, document),
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
