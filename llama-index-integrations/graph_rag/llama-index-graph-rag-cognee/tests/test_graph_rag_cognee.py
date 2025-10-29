import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import cognee
import pytest
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="mock strategy requires python3.10 or higher"
)
@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_graph_rag_cognee():
    documents = [
        Document(
            text="Jessica Miller, Experienced Sales Manager with a strong track record in driving sales growth and building high-performing teams."
        ),
        Document(
            text="David Thompson, Creative Graphic Designer with over 8 years of experience in visual design and branding."
        ),
    ]

    # Instantiate cognee GraphRAG
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="kuzu",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_db",
    )

    # Add data to cognee
    await cogneeRAG.add(documents, "test")
    # Process data into a knowledge graph
    await cogneeRAG.process_data("test")

    # Answer prompt based on knowledge graph
    search_results = await cogneeRAG.search("Tell me who are the people mentioned?")

    assert len(search_results) > 0, "No search results found"

    print("\n\nAnswer based on knowledge graph:\n")
    for result in search_results:
        print(f"{result}\n")

    # Answer prompt based on RAG
    search_results = await cogneeRAG.rag_search("Tell me who are the people mentioned?")

    assert len(search_results) > 0, "No search results found"

    print("\n\nAnswer based on RAG:\n")
    for result in search_results:
        print(f"{result}\n")

    # Search for related nodes
    search_results = await cogneeRAG.get_related_nodes("person")
    print("\n\nRelated nodes are:\n")
    for result in search_results:
        print(f"{result}\n")

    assert len(search_results) > 0, "No search results found"

    # Clean all data from previous runs
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_empty_documents():
    """Test handling of empty document lists and documents with empty text."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Test empty list
    with pytest.raises(
        ValueError,
        match="Invalid data type. Please provide a list of Documents or a single Document.",
    ):
        await cogneeRAG.add([])

    # Test list with empty text documents
    empty_docs = [Document(text=""), Document(text="   ")]
    await cogneeRAG.add(empty_docs)

    # Should not raise an exception
    assert True


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_single_document():
    """Test processing a single document vs a list of documents."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Test single document
    single_doc = Document(text="This is a single test document.")
    await cogneeRAG.add(single_doc)

    # Test list with single document
    doc_list = [Document(text="This is a document in a list.")]
    await cogneeRAG.add(doc_list)

    # Should not raise an exception
    assert True


@pytest.mark.asyncio
async def test_visualize_graph():
    """Test graph visualization functionality."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Mock the cognee visualization function
    with patch("cognee.visualize_graph", new_callable=AsyncMock) as mock_viz:
        # Test with default parameters
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = await cogneeRAG.visualize_graph(
                open_browser=False, output_file_path=temp_dir
            )
            assert result_path.endswith("graph_visualization.html")
            assert temp_dir in result_path
            mock_viz.assert_called_once()

    # Test with invalid directory
    with pytest.raises(ValueError, match="is not a directory"):
        await cogneeRAG.visualize_graph(
            output_file_path="/invalid/path/that/does/not/exist"
        )


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_search_error_handling():
    """Test error handling in search methods."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Mock cognee search to return empty results
    with patch("cognee.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        # Test empty search results
        results = await cogneeRAG.search("non-existent query")
        assert results == []

        results = await cogneeRAG.rag_search("non-existent query")
        assert results == []

        results = await cogneeRAG.get_related_nodes("non-existent-node")
        assert results == []


@pytest.mark.asyncio
async def test_mock_full_workflow():
    """Test complete workflow with mocked cognee functions for offline testing."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    documents = [
        Document(text="Test document about artificial intelligence."),
        Document(text="Machine learning is a subset of AI."),
    ]

    # Mock all cognee functions
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch("cognee.search", new_callable=AsyncMock) as mock_search,
        patch(
            "cognee.modules.users.methods.get_default_user", new_callable=AsyncMock
        ) as mock_user,
    ):
        # Setup mock returns
        mock_user.return_value = {"id": "test-user"}
        mock_search.return_value = ["Mocked search result"]

        # Test add documents
        await cogneeRAG.add(documents)
        mock_add.assert_called_once()

        # Test process data
        await cogneeRAG.process_data()
        mock_cognify.assert_called_once()

        # Test search methods
        search_results = await cogneeRAG.search("test query")
        assert len(search_results) == 1
        assert search_results[0] == "Mocked search result"

        rag_results = await cogneeRAG.rag_search("test query")
        assert len(rag_results) == 1

        related_results = await cogneeRAG.get_related_nodes("test_node")
        assert len(related_results) == 1


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_mixed_document_types():
    """Test handling of mixed valid and invalid document types."""
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    # Create mixed list with valid documents and edge cases
    mixed_docs = [
        Document(text="Valid document with content."),
        Document(text=""),  # Empty text
        Document(text="   "),  # Whitespace only
        Document(text="Another valid document."),
    ]

    # Should handle gracefully without exceptions
    await cogneeRAG.add(mixed_docs)
    assert True
