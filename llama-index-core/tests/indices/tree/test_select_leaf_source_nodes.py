"""
Regression test for https://github.com/run-llama/llama_index/issues/21441.

TreeSelectLeafRetriever._query() used to return Response(source_nodes=[]),
dropping all retrieval provenance. After the fix, the selected leaf nodes
should be surfaced via Response.source_nodes.
"""

from typing import Dict, Tuple

import pytest
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.tree.select_leaf_retriever import (
    TreeSelectLeafRetriever,
)
from llama_index.core.schema import Document


@pytest.fixture()
def single_doc():
    # Matches the fixture used in the other tree tests.
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    return [Document(text=doc_text)]


def _split_kwargs(struct_kwargs: Tuple[Dict, Dict]) -> Tuple[Dict, Dict]:
    index_kwargs, query_kwargs = struct_kwargs
    return index_kwargs, query_kwargs


def test_select_leaf_retriever_query_populates_source_nodes(
    single_doc,
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs,
) -> None:
    """_query() must return the selected leaf nodes as Response.source_nodes."""
    index_kwargs, query_kwargs = _split_kwargs(struct_kwargs)
    tree = TreeIndex.from_documents(single_doc, **index_kwargs)
    retriever = TreeSelectLeafRetriever(
        index=tree,
        child_branch_factor=1,
        **query_kwargs,
    )

    response = retriever._query(QueryBundle("What is?"))

    assert response.source_nodes, (
        "TreeSelectLeafRetriever._query() dropped source nodes (#21441)."
    )
    for node_with_score in response.source_nodes:
        assert node_with_score.node is not None
