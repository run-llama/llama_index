from typing import List

import pytest
from sqlalchemy import create_engine, text

from llama_index.readers.database import DatabaseReader
from llama_index.core.schema import Document


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sqlite_engine():
    """
    Return an in-memory SQLite engine using a URI that allows sharing
    the database across connections within the same process.
    """
    # This URI creates a named in-memory database that persists
    # as long as at least one connection is open, and is shareable.
    db_uri = (
        "sqlite:///file:llamaindex_reader_test_db?mode=memory&cache=shared&uri=true"
    )
    engine = create_engine(db_uri, future=True)

    # Set up schema + sample data (ensure clean state first)
    with engine.begin() as conn:
        # Drop table if it exists from a previous potentially failed run
        conn.execute(text("DROP TABLE IF EXISTS items"))
        # Create table (no schema prefix)
        conn.execute(
            text(
                """
                CREATE TABLE items (
                    id      INTEGER PRIMARY KEY,
                    name    TEXT,
                    value   INTEGER
                )
                """
            )
        )
        # Insert data (no schema prefix)
        conn.execute(
            text(
                """
                INSERT INTO items (name, value)
                VALUES ('foo', 10), ('bar', 20)
                """
            )
        )
    # The engine is now configured with a persistent in-memory DB
    # containing the 'items' table.
    return engine

    # Optional teardown: dispose engine if needed, though usually not required
    # engine.dispose()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _create_reader(engine):
    """Utility to build a DatabaseReader for the given engine."""
    return DatabaseReader(engine=engine)


def _get_all_docs(reader: DatabaseReader, **kwargs) -> List[Document]:
    """Convenience wrapper that returns a list of Documents."""
    return reader.load_data(
        query="SELECT id, name, value FROM items ORDER BY id",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_load_data_basic(sqlite_engine):
    """It should return two Document objects with concatenated text."""
    reader = _create_reader(sqlite_engine)
    docs = _get_all_docs(reader)

    assert len(docs) == 2
    assert docs[0].text_resource and docs[0].text_resource.text
    assert docs[0].text_resource.text.startswith("id: 1")
    assert docs[0].text_resource.text.endswith("value: 10")


def test_metadata_and_exclusion(sqlite_engine):
    """
    `metadata_cols` should be promoted to metadata and
    `excluded_text_cols` should remove columns from text.
    """
    reader = _create_reader(sqlite_engine)
    docs = _get_all_docs(
        reader,
        metadata_cols=[("id", "item_id"), "value"],
        excluded_text_cols=["value"],
    )

    doc = docs[0]
    # `value` excluded from text, included as metadata
    assert "value:" not in doc.text
    assert doc.metadata == {"item_id": 1, "value": 10}


def test_resource_id_fn(sqlite_engine):
    """Custom `document_id` should drive `doc_id`."""
    reader = _create_reader(sqlite_engine)
    docs = _get_all_docs(
        reader,
        document_id=lambda row: f"custom-{row['id']}",
    )

    assert docs[0].id_ == "custom-1"
    assert docs[1].id_ == "custom-2"


def test_lazy_load_data_generator(sqlite_engine):
    """`lazy_load_data` should yield Documents lazily."""
    reader = _create_reader(sqlite_engine)
    gen = reader.lazy_load_data(query="SELECT * FROM items")
    docs = list(gen)

    assert len(docs) == 2
    assert all(hasattr(d, "text_resource") for d in docs)
    assert all(hasattr(d.text_resource, "text") for d in docs)


@pytest.mark.asyncio
async def test_aload_data_async(sqlite_engine):
    """`aload_data` wraps the sync loader via asyncio.to_thread()."""
    reader = _create_reader(sqlite_engine)
    docs = await reader.aload_data(query="SELECT * FROM items")

    assert len(docs) == 2
    assert docs[0].text_resource and docs[0].text_resource.text
    assert docs[0].text_resource.text.startswith("id: 1")
