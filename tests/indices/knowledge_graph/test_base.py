"""Test knowledge graph index."""

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.query.knowledge_graph.query import GPTKGTableQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import (
    MOCK_KG_TRIPLET_EXTRACT_PROMPT,
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
)


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    # NOTE: in this unit test, document text == triplets
    doc_text = "(foo, is, bar)\n" "(hello, is not, world)\n" "(Jane, is mother of, Bob)"
    return [Document(doc_text)]


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "kg_triple_extract_template": MOCK_KG_TRIPLET_EXTRACT_PROMPT,
    }
    query_kwargs = {
        "query_keyword_extract_template": MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    }
    return index_kwargs, query_kwargs


def mock_extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    """Mock extract triplets."""
    lines = text.split("\n")
    triplets: List[Tuple[str, str, str]] = []
    for line in lines:
        tokens = line[1:-1].split(",")
        tokens = [t.strip() for t in tokens]

        subj, pred, obj = tokens
        triplets.append((subj, pred, obj))
    return triplets


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_build_kg(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    """Test build knowledge graph."""
    index = GPTKnowledgeGraphIndex(documents)
    # NOTE: in these unit tests, document text == triplets
    table_chunks = {n.text for n in index.index_struct.text_chunks.values()}
    assert len(table_chunks) == 3
    assert "(foo, is, bar)" in table_chunks
    assert "(hello, is not, world)" in table_chunks
    assert "(Jane, is mother of, Bob)" in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert index.index_struct.table.keys() == {
        "foo",
        "bar",
        "hello",
        "world",
        "Jane",
        "Bob",
    }


@patch_common
@patch.object(
    GPTKnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Any,
    documents: List[Document],
) -> None:
    """Test query."""
    index = GPTKnowledgeGraphIndex(documents)
    response = index.query("foo")
    # when include_text is True, the first node is the raw text
    assert str(response) == "foo:(foo, is, bar)"
    assert response.extra_info is not None
    assert response.extra_info["kg_rel_map"] == {
        "foo": [("bar", "is")],
    }

    # test specific query class
    query = GPTKGTableQuery(
        index.index_struct,
        llm_predictor=index.llm_predictor,
        prompt_helper=index.prompt_helper,
        docstore=index.docstore,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = query._get_nodes_for_response(query_bundle)
    assert nodes[0].get_text() == "(foo, is, bar)"
    assert (
        nodes[1].get_text() == "The following are knowledge triplets in the "
        "form of (subset, predicate, object):\n('foo', 'is', 'bar')"
    )

    query = GPTKGTableQuery(
        index.index_struct,
        llm_predictor=index.llm_predictor,
        prompt_helper=index.prompt_helper,
        docstore=index.docstore,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
        include_text=False,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = query._get_nodes_for_response(query_bundle)
    assert (
        nodes[0].get_text() == "The following are knowledge triplets in the form of "
        "(subset, predicate, object):\n('foo', 'is', 'bar')"
    )
