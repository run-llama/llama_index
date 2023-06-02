"""Test tree index."""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

from llama_index.data_structs.data_structs import IndexGraph
from llama_index.data_structs.node import Node
from llama_index.indices.service_context import ServiceContext
from llama_index.storage.docstore import BaseDocumentStore
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.readers.schema.base import Document


def _get_left_or_right_node(
    docstore: BaseDocumentStore,
    index_graph: IndexGraph,
    node: Optional[Node],
    left: bool = True,
) -> Node:
    """Get 'left' or 'right' node."""
    children_dict = index_graph.get_children(node)
    indices = list(children_dict.keys())
    index = min(indices) if left else max(indices)
    node_id = children_dict[index]
    return docstore.get_node(node_id)


def test_build_tree(
    documents: List[Document],
    mock_service_context: ServiceContext,
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(
        documents, service_context=mock_service_context, **index_kwargs
    )
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes

    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."
    assert nodes[4].text == ("Hello world.\nThis is a test.")
    assert nodes[5].text == ("This is another test.\nThis is a test v2.")

    # test ref doc info
    all_ref_doc_info = tree.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert documents[idx].doc_id == ref_doc_id


def test_build_tree_with_embed(
    documents: List[Document],
    mock_service_context: ServiceContext,
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text, embedding=[0.1, 0.2, 0.3])
    tree = GPTTreeIndex.from_documents(
        [document], service_context=mock_service_context, **index_kwargs
    )
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    all_nodes = tree.docstore.get_node_dict(tree.index_struct.all_nodes)
    assert all_nodes[0].text == "Hello world."
    assert all_nodes[1].text == "This is a test."
    assert all_nodes[2].text == "This is another test."
    assert all_nodes[3].text == "This is a test v2."
    # make sure all leaf nodes have embeddings
    for i in range(4):
        assert all_nodes[i].embedding == [0.1, 0.2, 0.3]
    assert all_nodes[4].text == ("Hello world.\nThis is a test.")
    assert all_nodes[5].text == ("This is another test.\nThis is a test v2.")


OUTPUTS = [
    ("Hello world.\nThis is a test.", ""),
    ("This is another test.\nThis is a test v2.", ""),
]


@patch("llama_index.indices.common_tree.base.run_async_tasks", side_effect=[OUTPUTS])
def test_build_tree_async(
    _mock_run_async_tasks: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
    struct_kwargs: Dict,
) -> None:
    """Test build tree with use_async."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(
        documents, use_async=True, service_context=mock_service_context, **index_kwargs
    )
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."
    assert nodes[4].text == ("Hello world.\nThis is a test.")
    assert nodes[5].text == ("This is another test.\nThis is a test v2.")


def test_build_tree_multiple(
    mock_service_context: ServiceContext,
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    new_docs = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(
        new_docs, service_context=mock_service_context, **index_kwargs
    )
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


def test_insert(
    documents: List[Document],
    mock_service_context: ServiceContext,
    struct_kwargs: Dict,
) -> None:
    """Test insert."""
    index_kwargs, _ = struct_kwargs
    tree = GPTTreeIndex.from_documents(
        documents, service_context=mock_service_context, **index_kwargs
    )

    # test insert
    new_doc = Document("This is a new doc.", doc_id="new_doc")
    tree.insert(new_doc)
    # Before:
    # Left root node: "Hello world.\nThis is a test."
    # "Hello world.", "This is a test" are two children of the left root node
    # After:
    # "Hello world.\nThis is a test\n.\nThis is a new doc." is the left root node
    # "Hello world", "This is a test\n.This is a new doc." are the children
    # of the left root node.
    # "This is a test", "This is a new doc." are the children of
    # "This is a test\n.This is a new doc."
    left_root = _get_left_or_right_node(tree.docstore, tree.index_struct, None)
    assert left_root.text == "Hello world.\nThis is a test."
    left_root2 = _get_left_or_right_node(tree.docstore, tree.index_struct, left_root)
    right_root2 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, left_root, left=False
    )
    assert left_root2.text == "Hello world."
    assert right_root2.text == "This is a test.\nThis is a new doc."
    left_root3 = _get_left_or_right_node(tree.docstore, tree.index_struct, right_root2)
    right_root3 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, right_root2, left=False
    )
    assert left_root3.text == "This is a test."
    assert right_root3.text == "This is a new doc."
    assert right_root3.ref_doc_id == "new_doc"

    # test insert from empty (no_id)
    tree = GPTTreeIndex.from_documents(
        [], service_context=mock_service_context, **index_kwargs
    )
    new_doc = Document("This is a new doc.")
    tree.insert(new_doc)
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert len(nodes) == 1
    assert nodes[0].text == "This is a new doc."

    # test insert from empty (with_id)
    tree = GPTTreeIndex.from_documents(
        [], service_context=mock_service_context, **index_kwargs
    )
    new_doc = Document("This is a new doc.", doc_id="new_doc_test")
    tree.insert(new_doc)
    assert len(tree.index_struct.all_nodes) == 1
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].text == "This is a new doc."
    assert nodes[0].ref_doc_id == "new_doc_test"


def test_twice_insert_empty(
    mock_service_context: ServiceContext,
) -> None:
    """# test twice insert from empty (with_id)"""
    tree = GPTTreeIndex.from_documents([], service_context=mock_service_context)

    # test first insert
    new_doc = Document("This is a new doc.", doc_id="new_doc")
    tree.insert(new_doc)
    # test second insert
    new_doc_second = Document("This is a new doc2.", doc_id="new_doc_2")
    tree.insert(new_doc_second)
    assert len(tree.index_struct.all_nodes) == 2


def _mock_tokenizer(text: str) -> int:
    """Mock tokenizer that splits by spaces."""
    return len(text.split(" "))
