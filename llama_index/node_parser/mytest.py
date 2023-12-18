import logging
import os
import sys
from datetime import datetime
import hashlib
import time
from typing import Dict

from typing import List, Optional

import llama_index
from llama_index.embeddings import CwEmbedding
from llama_index.llms import OpenAI, CwLM
from llama_index.node_parser.node_utils import docIdgen, build_nodes_from_splits_v2
from llama_index.readers.file.base import default_file_metadata_func
from llama_index import Document, VectorStoreIndex, ServiceContext, OpenAIEmbedding, StorageContext
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo, ObjectType, BaseNode
from llama_index.vector_stores.clickhouse import ClickhouseVectorStore, Record


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

    doc_id = docIdgen(version="", path=fpath)

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
    doc_metadata["doc_id"] = doc_id
    doc_metadata["doc_path"] = doc_metadata["file_path"]
    doc_metadata["doc_version"] = "1.0.0"
    doc_metadata["doc_author"] = "any"
    doc_metadata["doc_category"] = "law"
    doc_metadata["doc_owner"] = "awiss"
    # embeding策略，Document Only
    doc_metadata["embedding_meta_model"] = ""
    doc_metadata["embedding_meta_dimension"] = ""
    doc_metadata["embedding_meta_ctime"] = ""
    doc_metadata["embedding_meta_desc"] = ""
    doc_metadata["embedding_optional_1_meta_model"] = ""
    doc_metadata["embedding_optional_1_meta_dimension"] = ""
    doc_metadata["embedding_optional_1_meta_ctime"] = ""
    doc_metadata["embedding_optional_1_meta_desc"] = ""

    document = Document()
    ##document.node_id = id ## document 没有nodeID
    document.id_ = doc_id
    document.text = "我是个文档"
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
            next_node = gen_example_node(i + 2)
        if this_node == None:
            this_node = gen_example_node(i + 1)
        this_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document.node_id)
        this_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=prev_node.node_id)
        this_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=next_node.node_id)
        this_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_node.node_id)
        nodes.append(this_node)
        prev_node = this_node
        this_node = next_node
    return nodes


if __name__ == '__main__':
    sample_nodes = gen_pdf_sample_nodes(5)
    document = None
    for x in sample_nodes:
        print(x.id_)
        print(x.metadata)
        print(x.get_content())
        if x.get_type() == ObjectType.DOCUMENT:
            x.metadata["embedding_meta_model"] = OpenAIEmbedding().model_name
            x.metadata["embedding_meta_dimension"] = 1536
            x.metadata["embedding_meta_ctime"] = datetime.now().strftime("%Y-%m-%d")
            x.metadata["embedding_meta_desc"] = "default, 对text字段做embedding"
            document = x

    namespace_id = "ns_0518"
    collection_id = "tb_9172"
    table_name = namespace_id + "__" + collection_id

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    vector_store = ClickhouseVectorStore(host="10.178.13.203",
                                         port=18123,
                                         username="default",
                                         password="12345",
                                         database="new_test",
                                         table_name=table_name,
                                         )
    embed_model = CwEmbedding(
        url="http://10.178.15.11:30200/inference/default/cw-embedding/v1/embeddings",
        model_name="text-embedding-ada-002"
    )
    llm = CwLM(
        url="http://10.178.15.11:30200/inference/built-in/cw-llm/v1/chat/completions",
        model_name="llama2-cw-4k-20231108-v1-gptq-4bit"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(llm=llm,
                                                   embed_model=embed_model
                                                   )
    index = VectorStoreIndex([],
                             storage_context=storage_context,
                             service_context=service_context
                             )

    index.insert_nodes(sample_nodes)
    document.set_content(document.get_content()+" 我被更新了")
    print(document)
    index.update_ref_doc(document)


    #index.delete_ref_doc(ref_doc_id=document.node_id, delete_from_docstore=True)

    print("Fin")
