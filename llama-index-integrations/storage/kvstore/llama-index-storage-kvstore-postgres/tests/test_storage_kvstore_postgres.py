import pytest
from importlib.util import find_spec
from unittest.mock import MagicMock, patch
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.postgres import PostgresKVStore

no_packages = (
    find_spec("psycopg2") is None
    or find_spec("sqlalchemy") is None
    or find_spec("asyncpg") is None
)


def test_class():
    names_of_base_classes = [b.__name__ for b in PostgresKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_initialization():
    errors = []
    try:
        pgstore1 = PostgresKVStore(table_name="mytable")
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore2 = PostgresKVStore(
            table_name="mytable", connection_string="connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore3 = PostgresKVStore(
            table_name="mytable", async_connection_string="async_connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore4 = PostgresKVStore(
            table_name="mytable",
            connection_string="connection_string",
            async_connection_string="async_connection_string",
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    assert sum(errors) == 3
    assert pgstore4._engine is None
    assert pgstore4._async_engine is None


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_creation_uses_inspect_when_schema_does_not_exist():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_instance.connection.return_value = MagicMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = []

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        from llama_index.storage.kvstore.postgres.base import inspect

        inspect.assert_called_once_with(mock_session_instance.connection())
        mock_inspector.get_schema_names.assert_called_once()

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) == 1

        from sqlalchemy.schema import CreateSchema

        executed_statement = execute_calls[0][0][0]
        assert isinstance(executed_statement, CreateSchema)
        assert executed_statement.element == "test_schema"


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_creation_uses_inspect_when_schema_exists():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_instance.connection.return_value = MagicMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = ["test_schema"]

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        from llama_index.storage.kvstore.postgres.base import inspect

        inspect.assert_called_once_with(mock_session_instance.connection())
        mock_inspector.get_schema_names.assert_called_once()

        mock_session_instance.execute.assert_not_called()


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_put_all_uses_safe_insert():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )
        pgstore._connect()
        pgstore._is_initialized = True

        test_data = [("key1", {"value": "data1"}), ("key2", {"value": "data2"})]
        pgstore.put_all(test_data)

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) >= 1

        executed_statement = execute_calls[-1][0][0]
        assert hasattr(executed_statement, "compile")


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_aput_all_uses_safe_insert():
    import sqlalchemy
    from unittest.mock import AsyncMock

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = AsyncMock()
    mock_begin_ctx_manager = MagicMock()
    mock_begin_ctx_manager.__aenter__ = AsyncMock()
    mock_begin_ctx_manager.__aexit__ = AsyncMock()
    mock_session_instance.begin.return_value = mock_begin_ctx_manager

    mock_session_ctx_manager = MagicMock()
    mock_session_ctx_manager.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_ctx_manager.__aexit__ = AsyncMock(return_value=None)

    mock_async_session_factory = MagicMock(return_value=mock_session_ctx_manager)

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_async_session_factory),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )
        pgstore._connect()
        pgstore._is_initialized = True

        mock_session_instance.execute = AsyncMock()

        test_data = [("key1", {"value": "data1"}), ("key2", {"value": "data2"})]
        await pgstore.aput_all(test_data)

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) >= 1

        executed_statement = execute_calls[-1][0][0]
        assert hasattr(executed_statement, "compile")


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_name_with_special_characters():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = []

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        special_schema = "test'schema"
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name=special_schema,
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) == 1

        from sqlalchemy.schema import CreateSchema

        executed_statement = execute_calls[0][0][0]
        assert isinstance(executed_statement, CreateSchema)
