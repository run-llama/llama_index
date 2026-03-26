import os
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest
from llama_index.graph_rag.cognee import CogneeGraphRAG


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="mock strategy requires python3.10 or higher"
)
@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio
async def test_visualize_graph(monkeypatch):
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

    # Mock cognee's visualize_graph function
    mock_visualize = AsyncMock(return_value=None)

    import cognee

    monkeypatch.setattr(cognee, "visualize_graph", mock_visualize)

    # Test with custom output path
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = await cogneeRAG.visualize_graph(
            open_browser=False, output_file_path=temp_dir
        )

        # Verify the function was called
        mock_visualize.assert_called_once()

        # Verify the returned path is correct
        expected_path = os.path.join(temp_dir, "graph_visualization.html")
        assert result_path == expected_path

    # Test with default path (home directory)
    mock_visualize.reset_mock()
    result_path = await cogneeRAG.visualize_graph(open_browser=False)

    # Verify the function was called again
    mock_visualize.assert_called_once()

    # Verify the returned path points to home directory
    home_dir = os.path.expanduser("~")
    expected_path = os.path.join(home_dir, "graph_visualization.html")
    assert result_path == expected_path
