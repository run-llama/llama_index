"""Test tree index."""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.docstore import BaseDocumentStore


def _get_left_or_right_node(
    docstore: BaseDocumentStore,
    index_graph: IndexGraph,
    node: Optional[BaseNode],
    left: bool = True,
) -> BaseNode:
    """Get 'left' or 'right' node."""
    children_dict = index_graph.get_children(node)
    indices = list(children_dict.keys())
    index = min(indices) if left else max(indices)
    node_id = children_dict[index]
    return docstore.get_node(node_id)


def test_build_tree(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    tree = TreeIndex.from_documents(documents, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes

    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."
    assert nodes[4].get_content() == ("Hello world.\nThis is a test.")
    assert nodes[5].get_content() == ("This is another test.\nThis is a test v2.")

    # test ref doc info
    all_ref_doc_info = tree.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert documents[idx].doc_id == ref_doc_id


def test_build_tree_with_embed(
    documents: List[Document],
    struct_kwargs: Dict,
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test build tree."""
    index_kwargs, _ = struct_kwargs
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text, embedding=[0.1, 0.2, 0.3])
    tree = TreeIndex.from_documents([document], **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    all_nodes = tree.docstore.get_node_dict(tree.index_struct.all_nodes)
    assert all_nodes[0].get_content() == "Hello world."
    assert all_nodes[1].get_content() == "This is a test."
    assert all_nodes[2].get_content() == "This is another test."
    assert all_nodes[3].get_content() == "This is a test v2."
    # make sure all leaf nodes have embeddings
    for i in range(4):
        assert all_nodes[i].embedding == [0.1, 0.2, 0.3]
    assert all_nodes[4].get_content() == ("Hello world.\nThis is a test.")
    assert all_nodes[5].get_content() == ("This is another test.\nThis is a test v2.")


OUTPUTS = [
    ("Hello world.\nThis is a test.", ""),
    ("This is another test.\nThis is a test v2.", ""),
]


@patch(
    "llama_index.core.indices.common_tree.base.run_async_tasks",
    side_effect=[OUTPUTS],
)
def test_build_tree_async(
    _mock_run_async_tasks: Any,
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test build tree with use_async."""
    index_kwargs, _ = struct_kwargs
    tree = TreeIndex.from_documents(documents, use_async=True, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."
    assert nodes[4].get_content() == ("Hello world.\nThis is a test.")
    assert nodes[5].get_content() == ("This is another test.\nThis is a test v2.")


def test_build_tree_multiple(
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test build tree."""
    new_docs = [
        Document(text="Hello world.\nThis is a test."),
        Document(text="This is another test.\nThis is a test v2."),
    ]
    index_kwargs, _ = struct_kwargs
    tree = TreeIndex.from_documents(new_docs, **index_kwargs)
    assert len(tree.index_struct.all_nodes) == 6
    # check contents of nodes
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].get_content() == "Hello world."
    assert nodes[1].get_content() == "This is a test."
    assert nodes[2].get_content() == "This is another test."
    assert nodes[3].get_content() == "This is a test v2."


def test_insert(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Dict,
) -> None:
    """Test insert."""
    index_kwargs, _ = struct_kwargs
    tree = TreeIndex.from_documents(documents, **index_kwargs)

    # test insert
    new_doc = Document(text="This is a new doc.", id_="new_doc")
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
    assert left_root.get_content() == "Hello world.\nThis is a test."
    left_root2 = _get_left_or_right_node(tree.docstore, tree.index_struct, left_root)
    right_root2 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, left_root, left=False
    )
    assert left_root2.get_content() == "Hello world."
    assert right_root2.get_content() == "This is a test.\nThis is a new doc."
    left_root3 = _get_left_or_right_node(tree.docstore, tree.index_struct, right_root2)
    right_root3 = _get_left_or_right_node(
        tree.docstore, tree.index_struct, right_root2, left=False
    )
    assert left_root3.get_content() == "This is a test."
    assert right_root3.get_content() == "This is a new doc."
    assert right_root3.ref_doc_id == "new_doc"

    # test insert from empty (no_id)
    tree = TreeIndex.from_documents([], **index_kwargs)
    new_doc = Document(text="This is a new doc.")
    tree.insert(new_doc)
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert len(nodes) == 1
    assert nodes[0].get_content() == "This is a new doc."

    # test insert from empty (with_id)
    tree = TreeIndex.from_documents([], **index_kwargs)
    new_doc = Document(text="This is a new doc.", id_="new_doc_test")
    tree.insert(new_doc)
    assert len(tree.index_struct.all_nodes) == 1
    nodes = tree.docstore.get_nodes(list(tree.index_struct.all_nodes.values()))
    assert nodes[0].get_content() == "This is a new doc."
    assert nodes[0].ref_doc_id == "new_doc_test"


def test_twice_insert_empty(patch_llm_predictor, patch_token_text_splitter) -> None:
    """# test twice insert from empty (with_id)."""
    tree = TreeIndex.from_documents([])

    # test first insert
    new_doc = Document(text="This is a new doc.", id_="new_doc")
    tree.insert(new_doc)
    # test second insert
    new_doc_second = Document(text="This is a new doc2.", id_="new_doc_2")
    tree.insert(new_doc_second)
    assert len(tree.index_struct.all_nodes) == 2


def _mock_tokenizer(text: str) -> int:
    """Mock tokenizer that splits by spaces."""
    return len(text.split(" "))
