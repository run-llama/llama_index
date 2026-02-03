import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Mock dependencies before import
mock_core = MagicMock()
mock_base_spec = MagicMock()


class MockBaseToolSpec:
    def __init__(self, *args, **kwargs):
        pass


mock_base_spec.BaseToolSpec = MockBaseToolSpec

module_patches = {
    "inferedge_moss": MagicMock(),
    "llama_index.core": mock_core,
    "llama_index.core.tools": mock_core,
    "llama_index.core.tools.tool_spec": mock_core,
    "llama_index.core.tools.tool_spec.base": mock_base_spec,
}

# Apply patches before importing the unit under test
with patch.dict("sys.modules", module_patches):
    from llama_index.tools.moss.base import MossToolSpec


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.create_index = AsyncMock()
    client.load_index = AsyncMock()

    # Mock query return structure
    mock_doc = MagicMock()
    mock_doc.metadata = {"page": "33", "source": "mock_source"}
    mock_doc.score = 10.00
    mock_doc.text = "mock content"

    results = MagicMock()
    results.docs = [mock_doc]
    results.score = 10.00

    client.query = AsyncMock(return_value=results)
    return client


@pytest.mark.asyncio
async def test_index_docs(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    # Test indexing
    await spec.index_docs([])

    assert not spec._index_loaded
    mock_client.create_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_query(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    # Test query
    output = await spec.query("mock")

    # Verify load_index was called (since _index_loaded starts as False)
    mock_client.load_index.assert_awaited_once()

    # Verify query results formatting
    assert "10.00" in output
    assert "mock_source" in output
    assert "mock" in output
    assert "33" in output


def test_initialization_validation():
    # Synchronous test
    client = MagicMock()

    # Test invalid alpha
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        MossToolSpec(client, "test", alpha=1.5)

    # Test invalid top_k
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        MossToolSpec(client, "test", top_k=0)
