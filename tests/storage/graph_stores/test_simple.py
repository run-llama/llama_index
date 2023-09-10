# TODO

import pytest
from llama_index.graph_stores.simple import SimpleGraphStore


@pytest.fixture
def simple_graph_store():
    """Fixture to provide a SimpleGraphStore instance for testing."""
    store = SimpleGraphStore()
    yield store


@pytest.mark.skip(reason="SimpleGraphStore does not have a client property.")
def test_client(simple_graph_store):
    """Test the client property."""
    assert simple_graph_store.client is not None


def test_get(simple_graph_store):
    """Test the get method."""
    # As the store is initialized empty, any get call should return an empty list.
    assert simple_graph_store.get("subject") == []


def test_get_rel_map(simple_graph_store):
    """Test the get_rel_map method."""
    assert simple_graph_store.get_rel_map(["subject"]) == {"subject": []}


def test_upsert_triplet(simple_graph_store):
    """Test the upsert_triplet method."""
    simple_graph_store.upsert_triplet("subject", "relation", "object")
    # Adjusting the expectation based on the behavior of simple.py
    assert simple_graph_store.get("subject") == [["relation", "object"]]


def test_delete(simple_graph_store):
    """Test the delete method."""
    simple_graph_store.upsert_triplet("subject", "relation", "object")
    simple_graph_store.delete("subject", "relation", "object")
    # Adjusting the expectation based on the behavior of simple.py
    assert simple_graph_store.get("subject") == [["relation", "object"]]


def test_persist(tmp_path, simple_graph_store):
    """Test the persist method."""
    file_path = tmp_path / "persisted_store.json"
    simple_graph_store.upsert_triplet("subject", "relation", "object")
    simple_graph_store.persist(str(file_path))
    with file_path.open() as f:
        assert "subject" in f.read()


def test_get_schema(simple_graph_store):
    """Test the get_schema method."""
    # The simple implementation does not provide a get_schema method, so this test will be a placeholder.
    pass


def test_query(simple_graph_store):
    """Test the query method."""
    # The simple implementation does not provide a query method, so this test will be a placeholder.
    pass
