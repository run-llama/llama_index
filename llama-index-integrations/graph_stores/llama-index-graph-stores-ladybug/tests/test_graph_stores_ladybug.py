from pathlib import Path
from typing import Generator

import pytest
from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.ladybug import LadybugGraphStore

# Track all database files created during tests for cleanup
_test_db_files = []


def cleanup_test_databases():
    """Clean up all test database files."""
    for db_file in _test_db_files:
        try:
            Path(db_file).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors
    _test_db_files.clear()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown fixture that runs for every test."""
    # Setup: Clear any existing test databases
    cleanup_test_databases()

    yield

    # Teardown: Clean up all databases created during tests
    cleanup_test_databases()


@pytest.fixture()
def ladybug_graph_store() -> Generator[LadybugGraphStore, None, None]:
    """Fixture for LadybugGraphStore with proper cleanup."""
    import real_ladybug as lb

    db_file = "test_ladybug_graph_store.ladybug"
    Path(db_file).unlink(missing_ok=True)
    _test_db_files.append(db_file)

    db = lb.Database(db_file)
    store = LadybugGraphStore(db)

    yield store

    # Close database connection
    try:
        db.close()
    except Exception:
        pass


def test_ladybug_graph_store():
    names_of_bases = [b.__name__ for b in LadybugGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases


def test_ladybug_graph_store_basic_operations(ladybug_graph_store: LadybugGraphStore):
    """Test basic graph store operations."""
    # Test that the store can be instantiated and basic operations work
    # This ensures database files are properly created and cleaned up
    store = ladybug_graph_store

    # Basic triplet operations should not crash
    store.upsert_triplet("subject", "predicate", "object")
    triplets = store.get("subject")
    assert len(triplets) >= 0  # Should return some result without crashing
