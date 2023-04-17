from typing import List

from gpt_index.constants import DATA_KEY, DOCSTORE_KEY, INDEX_STRUCT_KEY, TYPE_KEY
from gpt_index.data_structs.data_structs import (
    KG,
    IndexDict,
    IndexGraph,
    IndexList,
    KeywordTable,
    Node,
    SimpleIndexDict,
)
from gpt_index.data_structs.data_structs_v2 import SimpleIndexDict as V2SimpleIndexDict
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.node_v2 import ImageNode as V2ImageNode
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.old_docstore import V1DocumentStore
from gpt_index.docstore import BaseDocumentStore as V2DocumentStore
from gpt_index.tools.migrate_v1_to_v2 import (
    convert_to_v2_dict,
    convert_to_v2_index_struct_and_docstore,
    index_dict_to_v2,
    index_graph_to_v2,
    index_list_to_v2,
    keyword_table_to_v2,
    kg_to_v2,
    node_to_v2,
)


def test_node_to_v2() -> None:
    """Test node conversion."""
    node_v1 = Node(
        text="test_text",
        ref_doc_id="test_ref_doc_id",
    )

    node_v2 = node_to_v2(node_v1)

    assert node_v2.text == node_v1.text
    assert node_v2.ref_doc_id == node_v1.ref_doc_id

    image_node_v1 = Node(
        text="test_text_2", ref_doc_id="test_ref_doc_id_2", image="test_image_str"
    )
    image_node_v2 = node_to_v2(image_node_v1)
    assert isinstance(image_node_v2, V2ImageNode)


def test_index_graph_to_v2() -> None:
    """Test index graph conversion."""
    struct_v1 = IndexGraph()

    root_node = Node(
        text="root_text",
        index=0,
        child_indices=set([1, 2]),
    )

    child_node_1 = Node(
        text="root_text",
        index=1,
        child_indices=set([3]),
    )

    child_node_2 = Node(
        text="root_text",
        index=2,
    )

    child_node_3 = Node(
        text="root_text",
        index=3,
    )

    all_nodes_list: List[Node] = [root_node, child_node_1, child_node_2, child_node_3]

    struct_v1.all_nodes = {node.index: node for node in all_nodes_list}
    struct_v1.root_nodes = {root_node.index: root_node}

    struct_v2, nodes_v2 = index_graph_to_v2(struct_v1)

    for node, node_v2 in zip(all_nodes_list, nodes_v2):
        children_dict = struct_v2.get_children(node_v2)
        assert set(children_dict.keys()) == node.child_indices


def test_index_list_to_v2() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    nodes = [node_1, node_2]
    struct_v1 = IndexList(nodes=nodes)
    struct_v2, _ = index_list_to_v2(struct_v1)

    assert len(struct_v2.nodes) == len(struct_v1.nodes)


def test_keyword_table_to_v2() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    struct_v1 = KeywordTable(
        table={"foo": set([1, 2]), "bar": set([1])},
        text_chunks={1: node_1, 2: node_2},
    )
    struct_v2, _ = keyword_table_to_v2(struct_v1)
    assert struct_v2.table["foo"] == set([node_1.get_doc_id(), node_2.get_doc_id()])
    assert struct_v2.table["bar"] == set([node_1.get_doc_id()])


def test_index_dict_to_v2() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    struct_v1 = IndexDict(
        nodes_dict={1: node_1, 2: node_2},
        id_map={
            "text_1": 1,
            "text_2": 2,
        },
        embeddings_dict={
            "node_id_1": [0.0],
            "node_id_2": [0.0],
        },
    )

    struct_v2, _ = index_dict_to_v2(struct_v1)
    int_to_text_id = {int_id: text_id for text_id, int_id in struct_v1.id_map.items()}
    expected_nodes_dict = {
        int_to_text_id[id_]: node.get_doc_id()
        for id_, node in struct_v1.nodes_dict.items()
    }
    assert struct_v2.nodes_dict == expected_nodes_dict

    simple_v1 = SimpleIndexDict(
        nodes_dict={1: node_1, 2: node_2},
        id_map={
            "text_1": 1,
            "text_2": 2,
        },
        embeddings_dict={
            "node_id_1": [0.0],
            "node_id_2": [0.0],
        },
    )
    simple_v2, _ = index_dict_to_v2(simple_v1)
    assert isinstance(simple_v2, V2SimpleIndexDict)


def test_kg_to_v2() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    struct_v1 = KG(
        table={"foo": set(["node_1", "node_2"]), "bar": set(["node_1"])},
        text_chunks={
            "node_1": node_1,
            "node_2": node_2,
        },
        rel_map={
            "apple": [("is", "sweet")],
        },
        embedding_dict={
            "embedding_key": [0.0],
        },
    )

    _, nodes = kg_to_v2(struct_v1)
    assert len(nodes) == len(struct_v1.text_chunks)


def test_convert_to_v2_index_struct_and_docstore() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    nodes = [node_1, node_2]
    struct_v1 = IndexList(nodes=nodes)
    docstore_v1 = V1DocumentStore()
    docstore_v1.add_documents([struct_v1])

    struct_v2, docstore_v2 = convert_to_v2_index_struct_and_docstore(
        struct_v1, docstore_v1
    )
    assert isinstance(struct_v2, V2IndexStruct)
    assert isinstance(docstore_v2, V2DocumentStore)

    assert len(docstore_v2.docs) == 2


def test_convert_to_v2_dict() -> None:
    node_1 = Node(
        text="test_text_1",
    )
    node_2 = Node(
        text="test_text_2",
    )
    nodes = [node_1, node_2]
    struct_v1 = IndexList(nodes=nodes)
    docstore_v1 = V1DocumentStore()
    docstore_v1.add_documents([struct_v1])

    v1_dict = {
        "index_struct_id": struct_v1.get_doc_id(),
        "docstore": docstore_v1.to_dict(),
    }

    v2_dict = convert_to_v2_dict(v1_dict)

    assert v2_dict[INDEX_STRUCT_KEY][TYPE_KEY] == IndexStructType.LIST
    assert DATA_KEY in v2_dict[INDEX_STRUCT_KEY]
    assert DOCSTORE_KEY in v2_dict
