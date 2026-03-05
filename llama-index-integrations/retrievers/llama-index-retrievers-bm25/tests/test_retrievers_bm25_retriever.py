import os
import shutil
from unittest.mock import MagicMock

import pytest

from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in BM25Retriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_scores():
    documents = [
        Document(text="Large Language Model"),
        Document(text="LlamaIndex is a data framework for your LLM application"),
        Document(text="How to use LlamaIndex"),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    result_nodes = retriever.retrieve("llamaindex llm")
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        assert node.score > 0.0


def test_large_value_of_top_k():
    documents = [Document.example()]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Passing a high value of similarity_top_k w.r.t Document example
    similarity_top_k = 20
    retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=similarity_top_k
    )
    result_nodes = retriever.retrieve("What is llama index about?")

    # As we had less nodes then similarity_top_k in the retriever
    assert len(result_nodes) < similarity_top_k
    # Retrieved nodes should be all the nodes added in retriever
    assert len(result_nodes) == len(nodes)


def test_metadata_filtering():
    # https://rachellegardner.com/the-one-sentence-summary/
    documents = [
        Document(
            text="A boy wizard begins training and must battle for his life with the Dark Lord who murdered his parents",
            metadata={
                "book": "Harry Potter And The Sorcerer's Stone",
                "author": "J.K. Rowling",
            },
        ),
        Document(
            text="In the south in the 1960s, three women cross racial boundaries to begin a movement that will forever change their town and the way women view one another.",
            metadata={"book": "The Help", "author": "Kathryn Stockett"},
        ),
        Document(
            text="Chaos is unleashed on a quiet coastal town when an unassuming crippled woman raises a young boy from the dead, unlocking a centuries-old curse.",
            metadata={"book": "When Faith Awakes", "author": "Mike Duran"},
        ),
        Document(
            text="Identity theft becomes fatal for a patient and puts a young doctor's reputation and medical practice in jeopardy",
            metadata={"book": "Medical Error", "author": "Richard Mabry"},
        ),
        Document(
            text="Harry returns to Hogwarts and uncovers the mystery behind a hidden chamber and a monster that is petrifying students, ultimately discovering that Tom Riddle—Voldemort's younger self—was behind it all.",
            metadata={
                "book": "Harry Potter and the Chamber of Secrets",
                "author": "J.K. Rowling",
            },
        ),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    similarity_top_k = 5
    without_filter_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=similarity_top_k
    )
    result_nodes = without_filter_retriever.retrieve(
        "Tell me about Harry potter books?"
    )
    # Only fetch those documents with score greater than 0
    relevant_nodes = _count_score_greater_than_zero(result_nodes)
    # As 5 documents it fetch all documents
    assert relevant_nodes == 5

    with_filter_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="author", operator=FilterOperator.EQ, value="J.K. Rowling"
                )
            ]
        ),
    )
    result_nodes = with_filter_retriever.retrieve("Tell me about Harry potter books?")
    # Only fetch those documents with score greater than 0
    relevant_nodes = _count_score_greater_than_zero(result_nodes)
    # It will fetch only filtered by metadata documents and others will be 0
    assert relevant_nodes == 2


def _count_score_greater_than_zero(nodes):
    count = 0
    for node in nodes:
        print(node.score)
        count = count + (node.score > 0)
    return count


def test_persist_and_load():
    documents = [Document.example()]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Passing a high value of similarity_top_k w.r.t Document example
    similarity_top_k = 20
    retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=similarity_top_k
    )

    # Persist the retriever
    try:
        retriever.persist("test_retriever")

        # Load the retriever
        _ = BM25Retriever.from_persist_dir("test_retriever")
    finally:
        # Clean up the test_retriever directory
        if os.path.exists("test_retriever"):
            shutil.rmtree("test_retriever")


def test_from_defaults_with_vector_store_that_stores_text():
    """Test BM25Retriever.from_defaults with stores_text vector stores."""
    # Build text nodes that would normally live in the vector store
    text_nodes = [
        TextNode(text="A hero fights an evil force", id_="1"),
        TextNode(text="A journey toward a specific goal", id_="2"),
    ]

    # Mock vector store: stores_text=True, get_nodes() returns nodes
    mock_vector_store = MagicMock()
    mock_vector_store.stores_text = True
    mock_vector_store.get_nodes.return_value = text_nodes

    # Mock docstore: empty (no docs stored)
    mock_docstore = MagicMock()
    mock_docstore.docs = {}

    # Mock VectorStoreIndex
    mock_index = MagicMock()
    mock_index.docstore = mock_docstore
    mock_index.vector_store = mock_vector_store

    retriever = BM25Retriever.from_defaults(index=mock_index, similarity_top_k=2)
    results = retriever.retrieve("hero")

    assert len(results) > 0
    mock_vector_store.get_nodes.assert_called_once()


def test_from_defaults_with_empty_vector_store_raises():
    """Test clear error when both docstore and vector store are empty."""
    mock_vector_store = MagicMock()
    mock_vector_store.stores_text = True
    mock_vector_store.get_nodes.return_value = []

    mock_docstore = MagicMock()
    mock_docstore.docs = {}

    mock_index = MagicMock()
    mock_index.docstore = mock_docstore
    mock_index.vector_store = mock_vector_store

    with pytest.raises(ValueError, match="No nodes found"):
        BM25Retriever.from_defaults(index=mock_index, similarity_top_k=2)


def test_from_defaults_with_vector_store_get_nodes_not_implemented():
    """Test fallback when vector store does not implement get_nodes."""
    mock_vector_store = MagicMock()
    mock_vector_store.stores_text = True
    mock_vector_store.get_nodes.side_effect = NotImplementedError

    mock_docstore = MagicMock()
    mock_docstore.docs = {}

    mock_index = MagicMock()
    mock_index.docstore = mock_docstore
    mock_index.vector_store = mock_vector_store

    with pytest.raises(ValueError, match="No nodes found"):
        BM25Retriever.from_defaults(index=mock_index, similarity_top_k=2)
