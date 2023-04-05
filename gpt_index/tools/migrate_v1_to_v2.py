"""Tool for migrating Index built with V1 data structs to V2."""
import dataclasses
import json
from typing import Dict, List, Optional, Tuple, Type

from gpt_index.constants import DOCSTORE_KEY, INDEX_STRUCT_KEY
from gpt_index.data_structs.table import SQLStructTable

try:
    import fire
except ImportError:
    print("Please run `pip install fire`")

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
    OpensearchIndexDict,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    WeaviateIndexDict,
)
from gpt_index.data_structs.data_structs_v2 import KG as V2KG
from gpt_index.data_structs.data_structs_v2 import ChromaIndexDict as V2ChromaIndexDict
from gpt_index.data_structs.data_structs_v2 import FaissIndexDict as V2FaissIndexDict
from gpt_index.data_structs.data_structs_v2 import IndexDict as V2IndexDict
from gpt_index.data_structs.data_structs_v2 import IndexGraph as V2IndexGraph
from gpt_index.data_structs.data_structs_v2 import IndexList as V2IndexList
from gpt_index.data_structs.data_structs_v2 import KeywordTable as V2KeywordTable
from gpt_index.data_structs.data_structs_v2 import (
    OpensearchIndexDict as V2OpensearchIndexDict,
)
from gpt_index.data_structs.data_structs_v2 import (
    PineconeIndexDict as V2PineconeIndexDict,
)
from gpt_index.data_structs.data_structs_v2 import QdrantIndexDict as V2QdrantIndexDict
from gpt_index.data_structs.data_structs_v2 import SimpleIndexDict as V2SimpleIndexDict
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.data_structs_v2 import (
    WeaviateIndexDict as V2WeaviateIndexDict,
)
from gpt_index.data_structs.node_v2 import DocumentRelationship
from gpt_index.data_structs.node_v2 import ImageNode as V2ImageNode
from gpt_index.data_structs.node_v2 import Node as V2Node
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.old_docstore import V1DocumentStore
from gpt_index.docstore import DocumentStore as V2DocumentStore
from gpt_index.tools.file_utils import add_prefix_suffix_to_file_path

INDEX_STRUCT_TYPE_TO_V1_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
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
    IndexStructType.NODE: Node,
}

V1_INDEX_STRUCT_KEY = "index_struct"
V1_INDEX_STRUCT_ID_KEY = "index_struct_id"
V1_DOC_STORE_KEY = "docstore"


def node_to_v2(node: Node) -> V2Node:
    if node.ref_doc_id is not None:
        relationships = {
            DocumentRelationship.SOURCE: node.ref_doc_id,
        }
    else:
        relationships = {}

    if node.image is None:
        return V2Node(
            text=node.text,
            doc_id=node.doc_id,
            embedding=node.embedding,
            doc_hash=node.doc_hash,
            extra_info=node.extra_info,
            node_info=node.node_info,
            relationships=relationships,
        )
    else:
        return V2ImageNode(
            text=node.text,
            doc_id=node.doc_id,
            embedding=node.embedding,
            doc_hash=node.doc_hash,
            extra_info=node.extra_info,
            node_info=node.node_info,
            image=node.image,
            relationships=relationships,
        )


def index_graph_to_v2(struct: IndexGraph) -> Tuple[V2IndexGraph, List[V2Node]]:
    all_nodes_v2 = {
        index: node.get_doc_id() for index, node in struct.all_nodes.items()
    }
    root_nodes_v2 = {
        index: node.get_doc_id() for index, node in struct.all_nodes.items()
    }
    node_id_to_children_ids_v2 = {}
    for node in struct.all_nodes.values():
        node_id = node.get_doc_id()
        children_ids = []
        for child_index in node.child_indices:
            child_id = struct.all_nodes[child_index].get_doc_id()
            children_ids.append(child_id)
        node_id_to_children_ids_v2[node_id] = children_ids
    struct_v2 = V2IndexGraph(
        all_nodes=all_nodes_v2,
        root_nodes=root_nodes_v2,
        node_id_to_children_ids=node_id_to_children_ids_v2,
    )

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
    nodes_v2 = [node_to_v2(node) for node in struct.text_chunks.values()]
    return struct_v2, nodes_v2


def index_dict_to_v2(struct: IndexDict) -> Tuple[V2IndexDict, List[V2Node]]:
    nodes_dict_v2 = {
        vector_id: struct.nodes_dict[int_id].get_doc_id()
        for vector_id, int_id in struct.id_map.items()
    }

    node_id_to_vector_id = {
        node_id: vector_id for vector_id, node_id in nodes_dict_v2.items()
    }
    doc_id_dict_v2: Dict[str, List[str]] = {}
    for node in struct.nodes_dict.values():
        node_id = node.get_doc_id()
        vector_id = node_id_to_vector_id[node_id]
        if node.ref_doc_id is not None:
            if node.ref_doc_id not in doc_id_dict_v2:
                doc_id_dict_v2[node.ref_doc_id] = []
            doc_id_dict_v2[node.ref_doc_id].append(vector_id)

    struct_v2 = V2IndexDict(
        nodes_dict=nodes_dict_v2,
        doc_id_dict=doc_id_dict_v2,
        embeddings_dict=struct.embeddings_dict,
    )
    nodes_v2 = [node_to_v2(node) for node in struct.nodes_dict.values()]

    if isinstance(struct, SimpleIndexDict):
        struct_v2 = V2SimpleIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, FaissIndexDict):
        struct_v2 = V2FaissIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, PineconeIndexDict):
        struct_v2 = V2PineconeIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, WeaviateIndexDict):
        struct_v2 = V2WeaviateIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, QdrantIndexDict):
        struct_v2 = V2QdrantIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, ChromaIndexDict):
        struct_v2 = V2ChromaIndexDict(**dataclasses.asdict(struct_v2))
    if isinstance(struct, OpensearchIndexDict):
        struct_v2 = V2OpensearchIndexDict(**dataclasses.asdict(struct_v2))
    return struct_v2, nodes_v2


def kg_to_v2(struct: KG) -> Tuple[V2KG, List[V2Node]]:
    struct_v2 = V2KG(
        table=struct.table,
        rel_map=struct.rel_map,
        embedding_dict=struct.embedding_dict,
    )
    nodes_v2 = [node_to_v2(node) for node in struct.text_chunks.values()]
    return struct_v2, nodes_v2


def convert_to_v2_index_struct_and_docstore(
    index_struct: IndexStruct, docstore: V1DocumentStore
) -> Tuple[V2IndexStruct, V2DocumentStore]:
    struct_v2: V2IndexStruct
    if isinstance(index_struct, IndexGraph):
        struct_v2, nodes_v2 = index_graph_to_v2(index_struct)
    elif isinstance(index_struct, IndexList):
        struct_v2, nodes_v2 = index_list_to_v2(index_struct)
    elif isinstance(index_struct, IndexDict):
        struct_v2, nodes_v2 = index_dict_to_v2(index_struct)
    elif isinstance(index_struct, KG):
        struct_v2, nodes_v2 = kg_to_v2(index_struct)
    elif isinstance(index_struct, KeywordTable):
        struct_v2, nodes_v2 = keyword_table_to_v2(index_struct)
    else:
        raise NotImplementedError(f"Cannot migrate {type(index_struct)} yet.")

    docstore_v2 = V2DocumentStore()
    docstore_v2.add_documents(nodes_v2, allow_update=False)
    return struct_v2, docstore_v2


def load_v1_index_struct_in_docstore(
    file_dict: dict,
) -> Tuple[IndexStruct, V1DocumentStore]:
    index_struct_id = file_dict[V1_INDEX_STRUCT_ID_KEY]
    docstore = V1DocumentStore.load_from_dict(
        file_dict[V1_DOC_STORE_KEY],
        type_to_struct=INDEX_STRUCT_TYPE_TO_V1_INDEX_STRUCT_CLASS,  # type: ignore
    )
    index_struct = docstore.get_document(index_struct_id)
    assert isinstance(index_struct, IndexStruct)
    return index_struct, docstore


def load_v1_index_struct_separate(
    file_dict: dict, index_struct_type: IndexStructType
) -> Tuple[IndexStruct, V1DocumentStore]:
    index_struct_cls = INDEX_STRUCT_TYPE_TO_V1_INDEX_STRUCT_CLASS[index_struct_type]
    index_struct = index_struct_cls.from_dict(file_dict[V1_INDEX_STRUCT_KEY])
    docstore = V1DocumentStore.load_from_dict(
        file_dict[V1_DOC_STORE_KEY],
        type_to_struct=INDEX_STRUCT_TYPE_TO_V1_INDEX_STRUCT_CLASS,  # type: ignore
    )
    return index_struct, docstore


def load_v1(
    file_dict: dict, index_struct_type: Optional[IndexStructType] = None
) -> Tuple[IndexStruct, V1DocumentStore]:
    if V1_INDEX_STRUCT_KEY in file_dict:
        assert index_struct_type is not None, "Must specify index_struct_type to load."
        index_struct, docstore = load_v1_index_struct_separate(
            file_dict, index_struct_type
        )
    elif V1_INDEX_STRUCT_ID_KEY in file_dict:
        index_struct, docstore = load_v1_index_struct_in_docstore(file_dict)
    else:
        raise ValueError("index_struct or index_struct_id must be provided.")
    return index_struct, docstore


def save_v2(index_struct: V2IndexStruct, docstore: V2DocumentStore) -> dict:
    return {
        INDEX_STRUCT_KEY: index_struct.to_dict(),
        DOCSTORE_KEY: docstore.serialize_to_dict(),
    }


def convert_to_v2_dict(
    v1_dict: dict, index_struct_type: Optional[IndexStructType] = None
) -> dict:
    index_struct, docstore = load_v1(v1_dict, index_struct_type)
    index_struct_v2, docstore_v2 = convert_to_v2_index_struct_and_docstore(
        index_struct, docstore
    )
    v2_dict = save_v2(index_struct_v2, docstore_v2)
    return v2_dict


def convert_to_v2_file(
    v1_path: str,
    index_struct_type: Optional[IndexStructType] = None,
    v2_path: Optional[str] = None,
    encoding: str = "ascii",
) -> None:
    with open(v1_path, "r") as f:
        file_str = f.read()
    v1_dict = json.loads(file_str)
    print(f"Successfully loaded V1 JSON file from: {v1_path}")

    v2_dict = convert_to_v2_dict(v1_dict, index_struct_type)

    v2_str = json.dumps(v2_dict)
    v2_path = v2_path or add_prefix_suffix_to_file_path(v1_path, suffix="_v2")
    with open(v2_path, "wt", encoding=encoding) as f:
        f.write(v2_str)
    print(f"Successfully created V2 JSON file at: {v2_path}")


if __name__ == "__main__":
    fire.Fire(convert_to_v2_file)
