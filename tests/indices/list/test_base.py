"""Test list index."""

import asyncio
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import patch

import pytest
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.list.embedding_query import GPTListIndexEmbeddingQuery
from gpt_index.indices.service_context import ServiceContext
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.node_parser.simple import SimpleNodeParser
from gpt_index.readers.schema.base import Document
from gpt_index.utils import globals_helper
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import (
    mock_llmpredictor_apredict,
    mock_llmpredictor_predict,
)
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
def test_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test list query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex.from_documents(documents, **index_kwargs)

    query_str = "What is?"
    response = index.query(query_str, mode="default", **query_kwargs)
    expected_answer = (
        "What is?:Hello world.:"
        "This is a test.:"
        "This is another test.:"
        "This is a test v2."
    )
    assert str(response) == expected_answer
    node_info = (
        response.source_nodes[0].node.node_info
        if response.source_nodes[0].node.node_info
        else {}
    )
    assert node_info["start"] == 0
    assert node_info["end"] == 12


@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "__init__", return_value=None)
def test_index_overlap(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    struct_kwargs: Dict,
) -> None:
    """Test node info calculation with overlapping node text."""
    index_kwargs, query_kwargs = struct_kwargs

    documents = [
        Document(
            "Hello world. This is a test 1. This is a test 2."
            + " This is a test 3. This is a test 4. This is a test 5.\n"
        )
    ]

    # A text splitter for test purposes
    _text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=30,
        chunk_overlap=10,
        tokenizer=globals_helper.tokenizer,
    )
    _node_parser = SimpleNodeParser(text_splitter=_text_splitter)
    service_context = ServiceContext.from_defaults(node_parser=_node_parser)

    index = GPTListIndex.from_documents(
        documents, service_context=service_context, **index_kwargs
    )

    query_str = "What is?"
    response = index.query(query_str, mode="default", **query_kwargs)
    node_info_0 = (
        response.source_nodes[0].node.node_info
        if response.source_nodes[0].node.node_info
        else {}
    )
    # First chunk: 'Hello world. This is a test 1. This is a test 2.
    # This is a test 3. This is a test 4. This is a'
    assert node_info_0["start"] == 0  # start at the start
    assert node_info_0["end"] == 94  # Length of first chunk.

    node_info_1 = (
        response.source_nodes[1].node.node_info
        if response.source_nodes[1].node.node_info
        else {}
    )
    # Second chunk: 'This is a test 4. This is a test 5.\n'
    assert node_info_1["start"] == 67  # Position of second chunk relative to start
    assert node_info_1["end"] == 103  # End index


@patch_common
def test_query_with_keywords(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test list query with keywords."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex.from_documents(documents, **index_kwargs)

    # test query with keywords
    query_str = "What is?"
    query_kwargs.update({"required_keywords": ["test"]})
    response = index.query(query_str, mode="default", **query_kwargs)
    expected_answer = (
        "What is?:This is a test.:" "This is another test.:" "This is a test v2."
    )
    assert str(response) == expected_answer

    query_kwargs.update({"exclude_keywords": ["test"], "required_keywords": []})
    response = index.query(query_str, mode="default", **query_kwargs)
    assert str(response) == ("What is?:Hello world.")


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
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex.from_documents(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(
        query_str, mode="embedding", similarity_top_k=1, **query_kwargs
    )
    assert str(response) == ("What is?:Hello world.")


@patch_common
def test_extra_info(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build/query with extra info."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    extra_info = {"extra_info": "extra_info", "foo": "bar"}
    new_document = Document(doc_text, extra_info=extra_info)
    list_index = GPTListIndex.from_documents([new_document])
    nodes = list_index.docstore.get_nodes(list_index.index_struct.nodes)
    assert nodes[0].get_text() == (
        "extra_info: extra_info\n" "foo: bar\n\n" "Hello world."
    )
    assert nodes[3].get_text() == (
        "extra_info: extra_info\n" "foo: bar\n\n" "This is a test v2."
    )


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


@patch_common
@patch.object(LLMPredictor, "apredict", side_effect=mock_llmpredictor_apredict)
def test_async_query(
    _mock_async_predict: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test list async query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTListIndex.from_documents(documents, **index_kwargs)

    # test default query mode.
    query_str = "What is?"
    task = index.aquery(query_str, mode="default", **query_kwargs)
    response = asyncio.run(task)
    expected_answer = (
        "What is?:Hello world.:"
        "This is a test.:"
        "This is another test.:"
        "This is a test v2."
    )
    assert str(response) == expected_answer
    node_info = (
        response.source_nodes[0].node.node_info
        if response.source_nodes[0].node.node_info
        else {}
    )
    assert node_info["start"] == 0
    assert node_info["end"] == 12

    # test tree_summarize mode
    query_str = "What is?"
    query_kwargs_copy = deepcopy(query_kwargs)
    query_kwargs_copy["response_mode"] = "tree_summarize"
    task = index.aquery(query_str, mode="default", **query_kwargs_copy)
    response = asyncio.run(task)
    assert str(response) == expected_answer
    node_info = (
        response.source_nodes[0].node.node_info
        if response.source_nodes[0].node.node_info
        else {}
    )
    assert node_info["start"] == 0
    assert node_info["end"] == 12
