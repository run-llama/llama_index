"""Test composing indices."""

from typing import Dict, List

from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.readers.schema.base import Document


def test_recursive_query_list_tree(
    documents: List[Document],
    mock_service_context: ServiceContext,
    index_kwargs: Dict,
) -> None:
    """Test query."""
    list_kwargs = index_kwargs["list"]
    tree_kwargs = index_kwargs["tree"]
    # try building a list for every two, then a tree
    list1 = GPTListIndex.from_documents(
        documents[0:2], service_context=mock_service_context, **list_kwargs
    )
    list2 = GPTListIndex.from_documents(
        documents[2:4], service_context=mock_service_context, **list_kwargs
    )
    list3 = GPTListIndex.from_documents(
        documents[4:6], service_context=mock_service_context, **list_kwargs
    )
    list4 = GPTListIndex.from_documents(
        documents[6:8], service_context=mock_service_context, **list_kwargs
    )

    summary1 = "summary1"
    summary2 = "summary2"
    summary3 = "summary3"
    summary4 = "summary4"
    summaries = [summary1, summary2, summary3, summary4]

    # there are two root nodes in this tree: one containing [list1, list2]
    # and the other containing [list3, list4]
    graph = ComposableGraph.from_indices(
        GPTTreeIndex,
        [
            list1,
            list2,
            list3,
            list4,
        ],
        index_summaries=summaries,
        service_context=mock_service_context,
        **tree_kwargs
    )
    assert isinstance(graph, ComposableGraph)
    query_str = "What is?"
    # query should first pick the left root node, then pick list1
    # within list1, it should go through the first document and second document
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == (
        "What is?:What is?:This is a test v2.:This is another test."
    )


def test_recursive_query_tree_list(
    documents: List[Document],
    mock_service_context: ServiceContext,
    index_kwargs: Dict,
) -> None:
    """Test query."""
    list_kwargs = index_kwargs["list"]
    tree_kwargs = index_kwargs["tree"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    tree1 = GPTTreeIndex.from_documents(
        documents[2:6], service_context=mock_service_context, **tree_kwargs
    )
    tree2 = GPTTreeIndex.from_documents(
        documents[:2] + documents[6:],
        service_context=mock_service_context,
        **tree_kwargs
    )
    summaries = [
        "tree_summary1",
        "tree_summary2",
    ]

    # there are two root nodes in this tree: one containing [list1, list2]
    # and the other containing [list3, list4]
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [tree1, tree2],
        index_summaries=summaries,
        service_context=mock_service_context,
        **list_kwargs
    )
    assert isinstance(graph, ComposableGraph)
    query_str = "What is?"
    # query should first pick the left root node, then pick list1
    # within list1, it should go through the first document and second document
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == (
        "What is?:What is?:This is a test.:What is?:This is a test v2."
    )


def test_recursive_query_table_list(
    documents: List[Document],
    mock_service_context: ServiceContext,
    index_kwargs: Dict,
) -> None:
    """Test query."""
    list_kwargs = index_kwargs["list"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    table1 = GPTSimpleKeywordTableIndex.from_documents(
        documents[4:6], service_context=mock_service_context, **table_kwargs
    )
    table2 = GPTSimpleKeywordTableIndex.from_documents(
        documents[2:3], service_context=mock_service_context, **table_kwargs
    )
    summaries = [
        "table_summary1",
        "table_summary2",
    ]

    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [table1, table2],
        index_summaries=summaries,
        service_context=mock_service_context,
        **list_kwargs
    )
    assert isinstance(graph, ComposableGraph)
    query_str = "World?"
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == ("World?:World?:Hello world.:None")

    query_str = "Test?"
    response = query_engine.query(query_str)
    assert str(response) == ("Test?:Test?:This is a test.:Test?:This is a test.")


def test_recursive_query_list_table(
    documents: List[Document],
    mock_service_context: ServiceContext,
    index_kwargs: Dict,
) -> None:
    """Test query."""
    list_kwargs = index_kwargs["list"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    list1 = GPTListIndex.from_documents(
        documents[0:2], service_context=mock_service_context, **list_kwargs
    )
    list2 = GPTListIndex.from_documents(
        documents[2:4], service_context=mock_service_context, **list_kwargs
    )
    list3 = GPTListIndex.from_documents(
        documents[4:6], service_context=mock_service_context, **list_kwargs
    )
    list4 = GPTListIndex.from_documents(
        documents[6:8], service_context=mock_service_context, **list_kwargs
    )
    summaries = [
        "foo bar",
        "apple orange",
        "toronto london",
        "cat dog",
    ]

    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [list1, list2, list3, list4],
        index_summaries=summaries,
        service_context=mock_service_context,
        **table_kwargs
    )
    assert isinstance(graph, ComposableGraph)
    query_str = "Foo?"
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is a test v2.:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.:Hello world.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is another test.:This is a test v2.")
