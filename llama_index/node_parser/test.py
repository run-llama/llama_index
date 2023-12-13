import llama_index
from llama_index.node_parser.node_utils import docIdgen, build_nodes_from_splits_v2
from llama_index.readers.file.base import default_file_metadata_func
from llama_index.vector_stores.clickhouse import Record
from llama_index import Document

doc_id = docIdgen("~/Desktop/园区大模型测试样例/统一转pdf/da/ada/a/da/fsf/s/dfsd/as/f/sdfs/f/saf/2")

print(doc_id)

doc_id_with_version = docIdgen(
    path="~/Desktop/园区大模型测试样例/统一转pdf/da/ada/a/da/fsf/s/dfsd/as/f/sdfs/f/saf/2",
    version="V1.0.0"
)

print(doc_id_with_version)

input_sample={}

##input_sample["id"]                                            #node_Id自动分配 Document类型，基于doc_id生成,不过规格要保持一致
input_sample["type"] = "TextNode" #单独成列                      #Document ImageDocument TextNode ImageNode IndexNode
input_sample["chunk_type"] = "BLOCK" #单独成列                   #BLOCK IMAGE DRAWING TOC ....
input_sample["text"] = "hello？"                                #单独成列
input_sample["hash"] = "hash_sha256"                              #普通基于text生成, Document 基于全文生成, 允许空

input_sample["doc_id"] = doc_id                                 #all must
input_sample["doc_path"] = "这个单独成列, 仅给Document Node"       #only doc
input_sample["doc_version"] = "这个单独成列, 仅给Document Node"    #only doc
input_sample["doc_author"] = "这个单独成列, 仅给Document Node"     #only doc
input_sample["doc_category"] = "这个单独成列, 仅给Document Node"   #only doc
input_sample["doc_owner"] = "这个是不是所有node都有单独成列??"       #only doc
input_sample["doc_mark_deleted"] = False                        #only doc

input_sample["abstract"] = "这个是不是所有node都有单独成列??"       #only doc
input_sample["keywords"] = "这个是不是所有node都有单独成列??"       #only doc



def chunked_doc_to_nodes(fpath, your_struct):
    metadata = default_file_metadata_func(fpath)   # 文档是这样，非文档类的自定义
    document = Document(text="", metadata=metadata)
    document.id_ = docIdgen(version="",path=fpath)
    document.metadata["额外的信息"] ="额外信息"
    document.metadata["额外的信息"] ="额外信息"
    document.metadata["额外的信息"] ="额外信息"
    document.metadata["额外的信息"] ="额外信息"
    document.metadata["额外的信息"] ="额外信息"
    document.hash = "搞一下"
    document.type_=""

    _splits=[]
    for xxx in your_struct:
        _dict ={}
        _dict["text"] ="Text字段,每个Node"
        _dict["Node额外信息"] ="额外信息"
        _dict["Node额外信息"] ="额外信息"
        _dict["Node额外信息"] ="额外信息"
        _dict["Node额外信息"] ="额外信息"
        _splits.append(_dict)

    nodes = build_nodes_from_splits_v2(all_splits=_splits,
                                       document=document
                                       )


if __name__ == '__main__':
    input_doc = input_sample.copy()
    input_doc["text"] = ""
    input_doc["hash"] = ""
    input_textNode_1 = input_sample.copy()

    recode  = Record(
        id=""
    )

    input_textNode_1["text"] = ""

    index = llama_index.VectorStoreIndex()
    index.




