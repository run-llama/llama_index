# TODO

import pytest
from llama_index.graph_stores.nebulagraph import NebulaGraphStore

# Placeholder for space_name. You might need to replace this with an actual space name in your NebulaGraph setup.
SPACE_NAME_PLACEHOLDER = "sample_space_name"


@pytest.fixture
def nebula_graph_store():
    """Fixture to provide a NebulaGraphStore instance for testing."""
    store = NebulaGraphStore(space_name=SPACE_NAME_PLACEHOLDER)
    yield store


def test_client(nebula_graph_store):
    """Test the client property."""
    assert nebula_graph_store.client is not None


def test_get(nebula_graph_store):
    """Test the get method."""
    result = nebula_graph_store.get("subject_placeholder")
    assert isinstance(result, list)


def test_get_rel_map(nebula_graph_store):
    """Test the get_rel_map method."""
    result = nebula_graph_store.get_rel_map(["subject_placeholder"])
    assert isinstance(result, dict)


def test_upsert_triplet(nebula_graph_store):
    """Test the upsert_triplet method."""
    nebula_graph_store.upsert_triplet(
        "subject_placeholder", "relation_placeholder", "object_placeholder")
    result = nebula_graph_store.get("subject_placeholder")
    assert result


def test_delete(nebula_graph_store):
    """Test the delete method."""
    nebula_graph_store.upsert_triplet(
        "subject_placeholder", "relation_placeholder", "object_placeholder")
    nebula_graph_store.delete("subject_placeholder",
                              "relation_placeholder", "object_placeholder")
    result = nebula_graph_store.get("subject_placeholder")
    assert not result


def test_persist(nebula_graph_store):
    """Test the persist method."""
    pass


def test_get_schema(nebula_graph_store):
    """Test the get_schema method."""
    result = nebula_graph_store.get_schema()
    assert isinstance(result, dict)


def test_query(nebula_graph_store):
    """Test the query method."""
    result = nebula_graph_store.query("sample_query_placeholder")
    assert result
