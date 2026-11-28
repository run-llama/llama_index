"""Offline tests for the engine / async_engine constructor arguments."""

from unittest.mock import patch

import pytest
from llama_index.vector_stores.openGauss import OpenGaussStore
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

PARAMS = {"table_name": "t", "embed_dim": 4, "perform_setup": False}


@pytest.fixture()
def engines():
    return create_engine("sqlite://"), create_async_engine("sqlite+aiosqlite://")


def test_engines_are_stored(engines) -> None:
    engine, async_engine = engines
    store = OpenGaussStore(engine=engine, async_engine=async_engine, **PARAMS)
    assert store._engine is engine
    assert store._async_engine is async_engine


def test_engines_default_to_none() -> None:
    store = OpenGaussStore(
        connection_string="opengauss+psycopg2://u:p@localhost/db",
        async_connection_string="opengauss+asyncpg://u:p@localhost/db",
        **PARAMS,
    )
    assert store._engine is None
    assert store._async_engine is None


def test_connect_reuses_supplied_engines(engines) -> None:
    engine, async_engine = engines
    store = OpenGaussStore(engine=engine, async_engine=async_engine, **PARAMS)

    with (
        patch("sqlalchemy.create_engine") as create_sync,
        patch("sqlalchemy.ext.asyncio.create_async_engine") as create_async,
    ):
        store._connect()

    create_sync.assert_not_called()
    create_async.assert_not_called()
    assert store._engine is engine
    assert store._async_engine is async_engine


@pytest.mark.parametrize("missing", ["engine", "async_engine"])
def test_only_one_engine_is_rejected(engines, missing: str) -> None:
    engine, async_engine = engines
    given = {"engine": engine, "async_engine": async_engine}
    given.pop(missing)
    with pytest.raises(ValueError, match="Both engine and async_engine"):
        OpenGaussStore(**given, **PARAMS)
