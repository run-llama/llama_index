"""General node utils."""

import logging
import os
from typing import List, Optional

from llama_index.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    NodeRelationship,
    TextNode, RelatedNodeInfo,
)
from llama_index.utils import truncate_text

import hashlib
import base64

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


"""
import Full_Info via Dict
step0: Prepare your chunked data. Not chunk by llama-index default policy.
step1: Build DocumentNode.
step2: traverse chunks, build TextNode
step3: return nodes
"""


def build_nodes_from_splits_v2(
        all_splits: List[dict[str]],
        document: BaseNode,  ## 这批预切数据对应的文档
        ref_doc: Optional[BaseNode] = None,  ## 文档抽象 源
) -> List[TextNode]:
    """Build nodes from pre chunked doc."""
    """Real Parser Service is independed from llama-index proj."""
    ref_doc = ref_doc or document

    # nodes: List[TextNode] = []
    nodes = []

    parent_node = None
    prev_node = None
    next_node = None

    for i, item in enumerate(all_splits):
        logger.debug(f"> Adding chunk: {truncate_text(item['text'], 50)}")
        if isinstance(document, ImageDocument):
            print("Can't handle ImageDocument")
        elif isinstance(document, Document):
            if parent_node == None and i == 0:
                print('遇到Documnet节点', document)
                parent_node = document
                prev_node = parent_node
                nodes.append(document)
                continue

            current_node = textNodeGen(item)
            print('遇到TextNode节点', current_node)
            if i < len(all_splits) - 1:
                next_node = textNodeGen(all_splits[i + 1])

            current_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document.node_id)
            current_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=prev_node.node_id)
            current_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=next_node.node_id)
            current_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_node.node_id)
            nodes.append(current_node)
            prev_node = current_node

        elif isinstance(document, TextNode):
            print("Not suitable")
        else:
            raise ValueError(f"Unknown document type: {type(document)}")

    return nodes


def textNodeGen(item: dict[str]):
    node_metadata = {}
    node_metadata["chunk_type"] = item["chunk_type"]
    node_metadata["deleted"] = item["deleted"]
    node_metadata["abstract"] = item["abstract"]
    node_metadata["keywords"] = item["keywords"]
    node_metadata["page_ids"] = item["page_ids"]

    node = TextNode()
    node.set_content(item["text"])
    node.hash = hashlib.sha256(node.text.encode("utf-8", "surrogatepass")).hexdigest()
    node.metadata = node_metadata
    return node


def docExist(document: BaseNode) -> bool:
    # TODO  may be better in other place
    return False


"""
doc_id generated by hash(version+path)
version means git proj's Version tags 
"""
def docIdGen(
        file_path: str,
        version: Optional[str] = ""):
    s = version + file_path
    # 使用 SHA256 算法
    hash_object = hashlib.sha256(s.encode("utf-8", "surrogatepass"))
    return hash_object.hexdigest()


def tableNameGen(
        namespace_id: str = "ns_id",
        collection_id: str = "cl_id"):
    return namespace_id + "__" + collection_id


def docHashGen(file_path):
    sha256_hash = hashlib.sha256()
    # 以二进制模式打开文件
    with open(file_path, "rb") as f:
        # 按块读取文件并更新哈希计算
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    # 返回16进制格式的哈希值
    return sha256_hash.hexdigest()
