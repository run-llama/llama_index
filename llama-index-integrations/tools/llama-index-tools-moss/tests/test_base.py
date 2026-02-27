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
    from llama_index.tools.moss.base import MossToolSpec, QueryOptions


def _make_mock_index(name: str, doc_count: int = 0, status: str = "ready") -> MagicMock:
    idx = MagicMock()
    idx.name = name
    idx.doc_count = doc_count
    idx.status = status
    return idx


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.create_index = AsyncMock()
    client.load_index = AsyncMock()
    client.delete_index = AsyncMock()

    # Mock query return structure
    mock_doc = MagicMock()
    mock_doc.metadata = {"page": "33", "source": "mock_source"}
    mock_doc.score = 10.00
    mock_doc.text = "mock content"

    results = MagicMock()
    results.docs = [mock_doc]
    results.score = 10.00

    client.query = AsyncMock(return_value=results)

    # Mock list_indexes return structure
    client.list_indexes = AsyncMock(
        return_value=[
            _make_mock_index("index_a", doc_count=5, status="ready"),
            _make_mock_index("index_b", doc_count=12, status="ready"),
        ]
    )

    return client


@pytest.mark.asyncio
async def test_index_docs(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    await spec.index_docs([])

    assert not spec._index_loaded
    mock_client.create_index.assert_awaited_once_with(
        "test", [], model_id="moss-minilm"
    )


@pytest.mark.asyncio
async def test_query(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    output = await spec.query("mock")

    # Verify load_index was called (since _index_loaded starts as False)
    mock_client.load_index.assert_awaited_once()

    # Verify query results formatting
    assert "10.00" in output
    assert "mock_source" in output
    assert "mock" in output
    assert "33" in output


@pytest.mark.asyncio
async def test_query_passes_options_to_client(mock_client):
    # MossQueryOptions in base.py is module_patches["inferedge_moss"].QueryOptions,
    # bound at import time when the patch was active.
    mock_query_options_cls = module_patches["inferedge_moss"].QueryOptions
    mock_query_options_cls.reset_mock()

    options = QueryOptions(top_k=7, alpha=0.3)
    spec = MossToolSpec(client=mock_client, index_name="test", query_options=options)

    await spec.query("something")

    # Verify MossQueryOptions was constructed with the values from QueryOptions
    mock_query_options_cls.assert_called_once_with(top_k=7, alpha=0.3)
    # Verify client.query received the constructed instance as its third arg
    call_args = mock_client.query.call_args
    assert call_args.args[0] == "test"
    assert call_args.args[1] == "something"


@pytest.mark.asyncio
async def test_query_skips_load_when_already_loaded(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")
    spec._index_loaded = True

    await spec.query("mock")

    mock_client.load_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_indexes(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    output = await spec.list_indexes()

    mock_client.list_indexes.assert_awaited_once()
    assert "index_a" in output
    assert "index_b" in output
    assert "5" in output
    assert "12" in output
    assert "ready" in output


@pytest.mark.asyncio
async def test_list_indexes_empty(mock_client):
    mock_client.list_indexes = AsyncMock(return_value=[])
    spec = MossToolSpec(client=mock_client, index_name="test")

    output = await spec.list_indexes()

    assert output == "No indexes found."


@pytest.mark.asyncio
async def test_delete_index(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")

    output = await spec.delete_index("other_index")

    mock_client.delete_index.assert_awaited_once_with("other_index")
    assert "other_index" in output
    # Deleting a different index should not reset _index_loaded
    assert not spec._index_loaded


@pytest.mark.asyncio
async def test_delete_current_index_resets_loaded_state(mock_client):
    spec = MossToolSpec(client=mock_client, index_name="test")
    spec._index_loaded = True

    await spec.delete_index("test")

    assert not spec._index_loaded


def test_query_options_application():
    client = MagicMock()
    options = QueryOptions(top_k=10, alpha=0.8, model_id="custom-model")
    spec = MossToolSpec(client=client, index_name="test", query_options=options)

    assert spec.top_k == 10
    assert spec.alpha == 0.8
    assert spec.model_id == "custom-model"


def test_initialization_validation():
    client = MagicMock()

    # Test invalid alpha
    opt1 = QueryOptions(alpha=2)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        MossToolSpec(client, "test", query_options=opt1)

    # Test invalid top_k
    opt2 = QueryOptions(top_k=-2)
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        MossToolSpec(client, "test", query_options=opt2)
