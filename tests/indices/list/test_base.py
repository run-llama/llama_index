"""Test list index."""

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.query.list.embedding_query import GPTListIndexEmbeddingQuery
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
    }
    query_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "refine_template": MOCK_REFINE_PROMPT,
    }
    return index_kwargs, query_kwargs


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@patch_common
def test_build_list(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
) -> None:
    """Test build list."""
    list_index = GPTListIndex(documents=documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."


@patch_common
def test_build_list_multiple(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
) -> None:
    """Test build list multiple."""
    documents = [
        Document("Hello world.\nThis is a test."),
        Document("This is another test.\nThis is a test v2."),
    ]
    list_index = GPTListIndex(documents=documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."


@patch_common
def test_list_insert(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
) -> None:
    """Test insert to list."""
    list_index = GPTListIndex([])
    assert len(list_index.index_struct.nodes) == 0
    list_index.insert(documents[0])
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."

    # test insert with ID
    document = documents[0]
    document.doc_id = "test_id"
    list_index = GPTListIndex([])
    list_index.insert(document)
    # check contents of nodes
    for node in list_index.index_struct.nodes:
        assert node.ref_doc_id == "test_id"


@patch_common
def test_list_delete(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    documents: List[Document],
) -> None:
    """Test insert to list and then delete."""
    new_documents = [
        Document("Hello world.\nThis is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]

    # delete from documents
    list_index = GPTListIndex(new_documents)
    # assert source doc is in docstore
    source_doc = list_index.docstore.get_document("test_id_1")
    assert source_doc is not None
    list_index.delete("test_id_1")
    assert len(list_index.index_struct.nodes) == 2
    assert list_index.index_struct.nodes[0].ref_doc_id == "test_id_2"
    assert list_index.index_struct.nodes[0].text == "This is another test."
    assert list_index.index_struct.nodes[1].ref_doc_id == "test_id_3"
    assert list_index.index_struct.nodes[1].text == "This is a test v2."
    # check that not in docstore anymore
    source_doc = list_index.docstore.get_document("test_id_1", raise_error=False)
    assert source_doc is None

    list_index = GPTListIndex(new_documents)
    list_index.delete("test_id_2")
    assert len(list_index.index_struct.nodes) == 3
    assert list_index.index_struct.nodes[0].ref_doc_id == "test_id_1"
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].ref_doc_id == "test_id_1"
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].ref_doc_id == "test_id_3"
    assert list_index.index_struct.nodes[2].text == "This is a test v2."


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
def test_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test list query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex(documents, **index_kwargs)

    query_str = "What is?"
    response = index.query(query_str, mode="default", **query_kwargs)
    assert str(response) == ("What is?:Hello world.")


@patch_common
def test_query_with_keywords(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test list query with keywords."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex(documents, **index_kwargs)

    # test query with keywords
    query_str = "What is?"
    query_kwargs.update({"required_keywords": ["test"]})
    response = index.query(query_str, mode="default", **query_kwargs)
    assert str(response) == ("What is?:This is a test.")

    query_kwargs.update({"exclude_keywords": ["Hello"]})
    response = index.query(query_str, mode="default", **query_kwargs)
    assert str(response) == ("What is?:This is a test.")


@patch_common
@patch.object(
    GPTListIndexEmbeddingQuery,
    "_get_embeddings",
    side_effect=_get_embeddings,
)
def test_embedding_query(
    _mock_similarity: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(
        query_str, mode="embedding", similarity_top_k=1, **query_kwargs
    )
    assert str(response) == ("What is?:Hello world.")
