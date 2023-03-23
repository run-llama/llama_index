"""Tool for migrating Index built with V1 data structs to V2."""
import json
from typing import Dict, List, Optional, Tuple, Type

try:
    import fire
except ImportError:
    print('Please run `pip install fire`')

from gpt_index.data_structs.data_structs import (
    KG,
    ChromaIndexDict,
    EmptyIndex,
    FaissIndexDict,
    IndexDict,
    IndexGraph,
    IndexList,
    IndexStruct,
    KeywordTable,
    Node,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    WeaviateIndexDict,
)
from gpt_index.data_structs.data_structs_v2 import IndexDict as V2IndexDict
from gpt_index.data_structs.data_structs_v2 import IndexGraph as V2IndexGraph
from gpt_index.data_structs.data_structs_v2 import IndexList as V2IndexList
from gpt_index.data_structs.data_structs_v2 import KeywordTable as V2KeywordTable
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.node_v2 import DocumentRelationship
from gpt_index.data_structs.node_v2 import Node as V2Node
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.data_structs.table_v2 import SQLStructTable
from gpt_index.docstore import DocumentStore
from gpt_index.indices.registry import INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS
from gpt_index.tools.file_utils import add_prefix_suffix_to_file_path

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, IndexStruct]= {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.SIMPLE_DICT: SimpleIndexDict,
    IndexStructType.DICT: FaissIndexDict,
    IndexStructType.WEAVIATE: WeaviateIndexDict,
    IndexStructType.PINECONE: PineconeIndexDict,
    IndexStructType.QDRANT: QdrantIndexDict,
    IndexStructType.CHROMA: ChromaIndexDict,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.KG: KG,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.NODE: Node
}

INDEX_STRUCT_KEY = "index_struct"
INDEX_STRUCT_ID_KEY = "index_struct_id"
DOC_STORE_KEY = "docstore"

def node_to_v2(node: Node) -> V2Node:
    return V2Node(
        text=node.text,
        doc_id=node.doc_id,
        embedding=node.embedding,
        doc_hash=node.doc_hash,
        extra_info=node.extra_info,
        node_info=node.node_info,
        image=node.image,
        relationships={
            DocumentRelationship.SOURCE: node.ref_doc_id,
        }
    )

def index_graph_to_v2(struct: IndexGraph) -> Tuple[V2IndexGraph, List[V2Node]]:
    all_nodes_v2 = {
        index: node.get_doc_id() for 
        index, node in struct.all_nodes.items()
    }
    root_nodes_v2 = {
        index: node.get_doc_id() for 
        index, node in struct.all_nodes.items()
    }
    node_id_to_children_ids_v2 = {}
    for node in struct.all_nodes.values():
        node_id = node.get_doc_id()
        children_ids = []
        for child_index in node.child_indices:
            child_id = struct.all_nodes[child_index].get_doc_id()
            children_ids.append(child_id)
        node_id_to_children_ids_v2[node_id] = children_ids
    struct_v2 = V2IndexGraph(all_nodes=all_nodes_v2, root_nodes=root_nodes_v2, node_id_to_children_ids=node_id_to_children_ids_v2)

    nodes_v2 = [node_to_v2(node) for node in struct.all_nodes.values()]
    return struct_v2, nodes_v2

def index_list_to_v2(struct: IndexList) -> Tuple[V2IndexList, List[V2Node]]:
    struct_v2 = V2IndexList(nodes=[node.get_doc_id() for node in struct.nodes])
    nodes_v2 = [node_to_v2(node) for node in struct.nodes]
    return struct_v2, nodes_v2

def keyword_table_to_v2(struct: KeywordTable) -> Tuple[V2KeywordTable, List[V2Node]]:
    table_v2 = {
        keyword: set(struct.text_chunks[index].get_doc_id() for index in indices)
        for keyword, indices in struct.table.items()
    }
    struct_v2 = V2KeywordTable(table=table_v2)
    nodes_v2 =  [node_to_v2(node) for node in struct.text_chunks.values()]
    return struct_v2, nodes_v2

def index_dict_to_v2(struct: IndexDict) -> Tuple[V2IndexDict]:
    nodes_dict_v2 = {
        vector_id: struct.nodes_dict[int_id].get_doc_id() for 
        vector_id, int_id in struct.id_map
    }

    node_id_to_vector_id = {
        node_id: vector_id for vector_id, node_id in nodes_dict_v2.items()
    }
    doc_id_dict_v2 = {}
    for node in struct.nodes_dict.values():
        node_id = node.get_doc_id()
        vector_id = node_id_to_vector_id[node_id]
        if node.ref_doc_id is not None:
            if node.ref_doc_id not in doc_id_dict_v2:
                doc_id_dict_v2[node.ref_doc_id] = []
            doc_id_dict_v2[node.ref_doc_id].append(vector_id)

    embeddings_dict_v2 = struct.embeddings_dict

    struct_v2 = V2IndexDict(nodes_dict=nodes_dict_v2, doc_id_dict=doc_id_dict_v2, embeddings_dict=embeddings_dict_v2)
    nodes_v2 = [node_to_v2(node) for node in node in struct.nodes_dict.values()]
    return struct_v2, nodes_v2

def convert_to_v2(index_struct: IndexStruct, docstore: DocumentStore) -> Tuple[V2IndexStruct, DocumentStore]:
    if isinstance(index_struct, IndexGraph):
        struct_v2, nodes_v2 = index_graph_to_v2(index_struct)
    elif isinstance(index_struct, IndexList):
        struct_v2, nodes_v2 = index_list_to_v2(index_struct)
    elif isinstance(index_struct, IndexDict):
        struct_v2, nodes_v2 = index_dict_to_v2(index_struct)
    else:
        raise NotImplementedError(f"Cannot migrate {type(index_struct)} yet.")
    
    docstore.add_documents(nodes_v2)
    return struct_v2, docstore
    

def load_v1_index_struct_in_docstore(file_dict):
    index_struct_id = file_dict[INDEX_STRUCT_ID_KEY]
    docstore = DocumentStore.load_from_dict(
        file_dict[DOC_STORE_KEY],
        type_to_struct=INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS
    )
    index_struct = docstore.get_document(index_struct_id)
    return index_struct, docstore


def load_v1_index_struct_separate(file_dict, index_struct_type: IndexStructType):
    index_struct_cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[index_struct_type]
    index_struct = index_struct_cls.from_dict(file_dict[INDEX_STRUCT_KEY])
    docstore = DocumentStore.load_from_dict(
        file_dict[DOC_STORE_KEY],
        type_to_struct=INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS
    )
    return index_struct, docstore


def load_v1(file_dict: dict, index_struct_type: Optional[IndexStructType] = None) -> Tuple[IndexStruct, DocumentStore]:
    if INDEX_STRUCT_KEY in file_dict:
        assert index_struct_type is not None, 'Must specify index_struct_type to load.'
        index_struct, docstore = load_v1_index_struct_separate(file_dict, index_struct_type)
    elif INDEX_STRUCT_ID_KEY in file_dict:
        index_struct, docstore = load_v1_index_struct_in_docstore(file_dict)
    else:
        raise ValueError("index_struct or index_struct_id must be provided.")
    return index_struct, docstore


def save_v2(index_struct: V2IndexStruct, docstore: DocumentStore) -> dict:
    return {
        INDEX_STRUCT_KEY: index_struct.to_dict(),
        DOC_STORE_KEY: docstore.serialize_to_dict(),
    }


def main(in_path: str, index_struct_type: Optional[IndexStructType]=None, out_path: Optional[str] = None, encoding: str = 'ascii'):
    with open(in_path, 'r') as f:
        file_str = f.read()
    file_dict = json.loads(file_str)
    print(f'Successfully loaded V1 JSON file from: {in_path}')

    index_struct, docstore = load_v1(file_dict, index_struct_type)
    index_struct_v2, docstore_v2 = convert_to_v2(index_struct, docstore)
    out_dict = save_v2(index_struct_v2, docstore_v2)
    
    out_str = json.dumps(out_dict)
    out_path = out_path or add_prefix_suffix_to_file_path(in_path, suffix='_v2')
    with open(out_path, "wt", encoding=encoding) as f:
        f.write(out_str)
    print(f'Successfully created V2 JSON file at: {out_path}')

if __name__ == '__main__':
    fire.Fire(main)