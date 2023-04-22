"""Test list index."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast

from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.common.base_retriever import BaseRetriever
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.schema import QueryMode
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common


@patch_common
def test_build_list(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build list."""
    list_index = GPTListIndex.from_documents(documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    node_ids = list_index.index_struct.nodes
    nodes = list_index.docstore.get_nodes(node_ids)
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


@patch_common
def test_refresh_list(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build list."""
    # add extra document
    more_documents = documents + [Document("Test document 2")]

    # ensure documents have doc_id
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)

    # create index
    list_index = GPTListIndex.from_documents(more_documents)

    # check that no documents are refreshed
    refreshed_docs = list_index.refresh(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is False

    # modify a document and test again
    more_documents = documents + [Document("Test document 2, now with changes!")]
    for i in range(len(more_documents)):
        more_documents[i].doc_id = str(i)

    # second document should refresh
    refreshed_docs = list_index.refresh(more_documents)
    assert refreshed_docs[0] is False
    assert refreshed_docs[1] is True

    test_node = list_index.docstore.get_node(list_index.index_struct.nodes[-1])
    assert test_node.text == "Test document 2, now with changes!"


@patch_common
def test_build_list_multiple(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test build list multiple."""
    documents = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    list_index = GPTListIndex.from_documents(documents)
    assert len(list_index.index_struct.nodes) == 4
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


@patch_common
def test_list_insert(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test insert to list."""
    list_index = GPTListIndex([])
    assert len(list_index.index_struct.nodes) == 0
    list_index.insert(documents[0])
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."

    # test insert with ID
    document = documents[0]
    document.doc_id = "test_id"
    list_index = GPTListIndex([])
    list_index.insert(document)
    # check contents of nodes
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    # check contents of nodes
    for node in nodes:
        assert node.ref_doc_id == "test_id"


@patch_common
def test_list_delete(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test insert to list and then delete."""
    new_documents = [
        Document("Hello world.\nThis is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]

    # delete from documents
    list_index = GPTListIndex.from_documents(new_documents)
    list_index.delete("test_id_1")
    assert len(list_index.index_struct.nodes) == 2
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_2"
    assert nodes[0].text == "This is another test."
    assert nodes[1].ref_doc_id == "test_id_3"
    assert nodes[1].text == "This is a test v2."
    # check that not in docstore anymore
    source_doc = list_index.docstore.get_document("test_id_1", raise_error=False)
    assert source_doc is None

    list_index = GPTListIndex.from_documents(new_documents)
    list_index.delete("test_id_2")
    assert len(list_index.index_struct.nodes) == 3
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    assert nodes[0].ref_doc_id == "test_id_1"
    assert nodes[0].text == "Hello world."
    assert nodes[1].ref_doc_id == "test_id_1"
    assert nodes[1].text == "This is a test."
    assert nodes[2].ref_doc_id == "test_id_3"
    assert nodes[2].text == "This is a test v2."


def _get_embeddings(
    query_str: str, nodes: List[Node]
) -> Tuple[List[float], List[List[float]]]:
    """Get node text embedding similarity."""
    text_embed_map: Dict[str, List[float]] = {
        "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
        "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
        "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
        "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
    }
    node_embeddings = []
    for node in nodes:
        node_embeddings.append(text_embed_map[node.get_text()])

    return [1.0, 0, 0, 0, 0], node_embeddings


@patch_common
def test_to_from_disk(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test saving to disk and from disk."""
    list_index = GPTListIndex.from_documents(documents)
    with TemporaryDirectory() as tmp_dir:
        list_index.save_to_disk(str(Path(tmp_dir) / "tmp.json"))
        new_list_index = cast(
            GPTListIndex, GPTListIndex.load_from_disk(str(Path(tmp_dir) / "tmp.json"))
        )
        assert len(new_list_index.index_struct.nodes) == 4
        nodes = new_list_index.docstore.get_nodes(new_list_index.index_struct.nodes)
        # check contents of nodes
        assert nodes[0].text == "Hello world."
        assert nodes[1].text == "This is a test."
        assert nodes[2].text == "This is another test."
        assert nodes[3].text == "This is a test v2."


@patch_common
def test_to_from_string(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test saving to disk and from disk."""
    list_index = GPTListIndex.from_documents(documents)
    new_list_index = cast(
        GPTListIndex, GPTListIndex.load_from_string(list_index.save_to_string())
    )
    assert len(new_list_index.index_struct.nodes) == 4
    nodes = new_list_index.docstore.get_nodes(new_list_index.index_struct.nodes)

    # check contents of nodes
    assert nodes[0].text == "Hello world."
    assert nodes[1].text == "This is a test."
    assert nodes[2].text == "This is another test."
    assert nodes[3].text == "This is a test v2."


def test_as_retriever(
    list_index: GPTListIndex,
) -> None:
    default_retriever = list_index.as_retriever(mode=QueryMode.DEFAULT)
    assert isinstance(default_retriever, BaseRetriever)

    embedding_retriever = list_index.as_retriever(mode=QueryMode.EMBEDDING)
    assert isinstance(embedding_retriever, BaseRetriever)
