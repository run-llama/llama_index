from datetime import datetime
import hashlib
import time
from typing import Dict

from typing import List, Optional

import llama_index
from llama_index.node_parser.node_utils import docIdgen, build_nodes_from_splits_v2
from llama_index.readers.file.base import default_file_metadata_func
from llama_index import Document, VectorStoreIndex
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo, ObjectType, BaseNode

print("hello")

doc_id = docIdgen("~/Desktop/园区大模型测试样例/统一转pdf/da/ada/a/da/fsf/s/dfsd/as/f/sdfs/f/saf/2")

print(doc_id)

doc_id_with_version = docIdgen(
    path="~/Desktop/园区大模型测试样例/统一转pdf/da/ada/a/da/fsf/s/dfsd/as/f/sdfs/f/saf/2",
    version="V1.0.0"
)

print(doc_id_with_version)



def gen_example_node(seqN: int):
    node_metadata = {}

    # 新增自定义 通用
    node_metadata["chunk_type"] = "BLOCK"
    node_metadata["deleted"] = False
    node_metadata["abstract"] = f"中国最新刑法典第{seqN}条"
    node_metadata["keywords"] = "中国，刑法"

    node = TextNode()

    node.set_content(f"我是Node_{seqN}")
    node.hash = hashlib.sha256(node.text.encode("utf-8", "surrogatepass")).hexdigest()
    node.metadata = node_metadata

    return node

def gen_pdf_sample_nodes(nodesNums: Optional[int] = 5) -> List[BaseNode]:
    fpath = "data/chinese_low/《中华人名共和国刑法典》_2023.pdf"
    # metadata = default_file_metadata_func(fpath)   # 文档是这样,，非文档类的自定义

    id = docIdgen(version="", path=fpath)
    print(id)
    doc_metadata = {}
    # 默认爬取
    doc_metadata["file_path"] = "data/chinese_low/《中华人名共和国刑法典》_2023.pdf"
    doc_metadata["file_name"] = "《中华人名共和国刑法典》_2023.pdf"
    doc_metadata["file_type"] = "pdf"
    doc_metadata["file_size"] = 4096
    doc_metadata["creation_date"] = datetime.now().strftime("%Y-%m-%d")
    doc_metadata["last_modified_date"] = datetime.now().strftime("%Y-%m-%d")
    doc_metadata["last_accessed_date"] = datetime.now().strftime("%Y-%m-%d")

    # 新增自定义 通用
    doc_metadata["type"] = "Document"
    doc_metadata["chunk_type"] = "Document"
    doc_metadata["deleted"] = False
    doc_metadata["abstract"] = "中国最新刑法典"
    doc_metadata["keywords"] = "中国，刑法"
    # 新增自定义 Document Only
    doc_metadata["doc_id"] = "pdf"
    doc_metadata["doc_path"] = doc_metadata["file_path"]
    doc_metadata["doc_version"] = "1.0.0"
    doc_metadata["doc_author"] = "any"
    doc_metadata["doc_category"] = "law"
    doc_metadata["doc_owner"] = "awiss"
    # embeding策略，Document Only
    doc_metadata["embedding_default_meta_model"] = ""
    doc_metadata["embedding_default_meta_dimension"] = ""
    doc_metadata["embedding_default_meta_ctime"] = ""
    doc_metadata["embedding_optional_1_meta_model"] = ""
    doc_metadata["embedding_optional_1_meta_dimension"] = ""
    doc_metadata["embedding_optional_1_meta_ctime"] = ""

    document = Document()
    ##document.node_id = id ## document 没有nodeID
    document.id_ = id
    document.text = ""
    document.hash = "d10a79a3408bd9ff8f5bae43284258a43f9741aea3a45e83ca9b773abca1b50e"
    document.metadata = doc_metadata

    ##################################################################################
    nodes = []
    nodes.append(document)

    parent_node = document
    prev_node = parent_node
    next_node = None
    this_node = None
    for i in range(nodesNums):
        if i < nodesNums:
            next_node = gen_example_node(i+1)
        if this_node == None:
            this_node = gen_example_node(i)
        this_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=prev_node.node_id)
        this_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=next_node.node_id)
        this_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_node.node_id)
        nodes.append(this_node)
        prev_node = this_node
        this_node = next_node
    return nodes


if __name__ == '__main__':
    print("hello")
    #print(gen_pdf_sample_nodes())