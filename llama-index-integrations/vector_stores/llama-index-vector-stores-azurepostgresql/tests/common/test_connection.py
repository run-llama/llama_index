"""Synchronous connection handling tests for Azure Database for PostgreSQL."""

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from typing import Any

import pytest
from psycopg import Connection, sql
from pydantic import BaseModel, ConfigDict

from llama_index.vector_stores.azure_postgres.common import (
    Extension,
    check_connection,
    create_extensions,
)


class MockCursorBase(BaseModel):
    """A minimal mock cursor base model used for testing DB interactions.

    Attributes:
        broken (bool): If True, simulates a broken cursor that fails queries.
        last_query (str | sql.SQL | None): Stores the last executed query for inspection.
        response (dict | None): Value to return from fetchone() when appropriate.
    """

    broken: bool = False
    last_query: str | sql.SQL | None = None
    response: dict | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class MockCursor(MockCursorBase):
    """A mock cursor implementing execute and fetchone for tests.

    The mock cursor records the last executed query and returns canned
    responses from the ``response`` attribute. When ``broken`` is True,
    ``fetchone`` returns None to simulate failures.
    """

    def execute(self, query: str | sql.SQL, _params=None) -> None:
        """Execute a SQL query and record it for later inspection."""
        self.last_query = query

    def fetchone(self) -> None | dict:
        """Return a single-row result dict."""
        assert self.last_query is not None, "No query executed."

        # We either give `"select 1"` or `sql.SQL(...)` as the last query.
        if isinstance(self.last_query, str):
            return None if self.broken else {"?column?": 1}

        return self.response


@pytest.fixture
def mock_cursor(
    connection: Connection,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    """Pytest fixture that replaces a real DB cursor with a MockCursor.

    Expects the parameterization to pass an instance of ``MockCursor``
    via ``request.param``. The fixture monkeypatches the connection's
    ``cursor`` method to return the supplied mock cursor as a context
    manager.
    """
    assert isinstance(request.param, MockCursor), "Expected a MockCursor instance."

    @contextmanager
    def mock_cursor(**_kwargs):
        yield request.param

    monkeypatch.setattr(connection, "cursor", mock_cursor)


class TestCheckConnection:
    """Tests for verifying the database connection and required extensions.

    These tests exercise ``check_connection`` with various mocked cursor
    responses to validate behavior for installed extensions, missing
    extensions, version mismatches, and broken cursors.
    """

    def test_it_works(self, connection: Connection) -> None:
        """Ensure ``check_connection`` returns None on a healthy connection."""
        assert check_connection(connection) is None

    @pytest.mark.parametrize(
        ["extension", "mock_cursor", "expected_result"],
        [
            (
                Extension(ext_name="test_ext", ext_version="1.0", schema_name="public"),
                MockCursor(
                    broken=False,
                    response={
                        "ext_name": "test_ext",
                        "ext_version": "1.0",
                        "schema_name": "public",
                    },
                ),
                nullcontext(None),
            ),
            (
                Extension(ext_name="test_ext", ext_version="1.0", schema_name="public"),
                MockCursor(broken=True, response=None),
                pytest.raises(AssertionError, match="Connection check failed"),
            ),
            (
                Extension(ext_name="test_ext", ext_version="1.0", schema_name="public"),
                MockCursor(broken=False, response=None),
                pytest.raises(
                    RuntimeError,
                    match="Required extension 'test_ext' is not installed.",
                ),
            ),
            (
                Extension(ext_name="test_ext", ext_version="1.0", schema_name="public"),
                MockCursor(broken=False, response={"ext_version": "wrong_version"}),
                pytest.raises(
                    RuntimeError,
                    match="Required extension 'test_ext' version mismatch: expected 1.0, got wrong_version.",
                ),
            ),
            (
                Extension(ext_name="test_ext", ext_version="1.0", schema_name="public"),
                MockCursor(
                    broken=False,
                    response={"ext_version": "1.0", "schema_name": "wrong_schema"},
                ),
                pytest.raises(
                    RuntimeError,
                    match="Required extension 'test_ext' is not installed in the expected schema: expected public, got wrong_schema.",
                ),
            ),
        ],
        ids=[
            "extension-installed",
            "broken-cursor",
            "extension-not-installed",
            "version-mismatch",
            "schema-mismatch",
        ],
        indirect=["mock_cursor"],
    )
    def test_mock_it_works(
        self,
        connection: Connection,
        extension: Extension,
        mock_cursor,
        expected_result: nullcontext | pytest.RaisesExc,
    ) -> None:
        """Run parameterized checks of ``check_connection`` using mocked cursors.

        Parameterization covers installed extension, broken cursor,
        missing extension, version mismatch, and schema mismatch cases.
        """
        with expected_result as e:
            assert check_connection(connection, required_extensions=[extension]) == e


@pytest.fixture
def extension_creatable(
    connection: Connection, request: pytest.FixtureRequest
) -> Generator[Extension, Any, None]:
    """Fixture that attempts to create (and later drop) a DB extension.

    Uses the provided ``Extension`` instance via ``request.param`` and
    will skip the test if creation fails. After the test, the extension
    is dropped if it was not previously installed.
    """
    assert isinstance(request.param, Extension), "Expected an Extension instance."

    ext_already_installed = False

    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                select extname, extversion
                  from pg_extension
                 where extname = %(ext_name)s
                """
            ),
            {"ext_name": request.param.ext_name},
        )
        result = cursor.fetchone()
        ext_already_installed = result is not None

        try:
            cursor.execute(
                sql.SQL(
                    """
                    create extension  if not exists {ext_name}
                                with  {schema_expr}
                                      {version_expr}
                                      {cascade_expr}
                    """
                ).format(
                    ext_name=sql.Identifier(request.param.ext_name),
                    schema_expr=sql.SQL("schema {schema_name}").format(
                        schema_name=sql.Identifier(request.param.schema_name)
                    )
                    if request.param.schema_name
                    else sql.SQL(""),
                    version_expr=sql.SQL("version {ext_version}").format(
                        ext_version=sql.Literal(request.param.ext_version)
                    )
                    if request.param.ext_version
                    else sql.SQL(""),
                    cascade_expr=sql.SQL("cascade")
                    if request.param.cascade
                    else sql.SQL(""),
                )
            )
        except Exception as e:
            pytest.skip(
                reason=f"Extension {request.param.ext_name} could not be created: {e}"
            )

    yield request.param

    if not ext_already_installed:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    drop extension if exists {ext_name}
                    """
                ).format(
                    ext_name=sql.Identifier(request.param.ext_name),
                )
            )


class TestCreateExtensions:
    """Tests that validate creating and handling of Postgres extensions.

    - ``test_it_works`` verifies that a valid extension can be created.
    - ``test_it_fails`` ensures that attempting to create a non-existent
      extension raises an informative exception.
    """

    @pytest.mark.parametrize(
        "extension_creatable",
        [Extension(ext_name="vector")],
        ids=["vector"],
        indirect=True,
    )
    def test_it_works(self, connection: Connection, extension_creatable: Extension):
        """Assert that creating a valid extension returns None (no error)."""
        assert (
            create_extensions(
                connection,
                required_extensions=[extension_creatable],
            )
            is None
        )

    def test_it_fails(self, connection: Connection):
        """Verify that creating a missing extension raises an exception."""
        extension = Extension(
            ext_name="non_existent_ext",
            ext_version="1.0",
            schema_name="public",
        )
        with pytest.raises(
            Exception, match='extension "non_existent_ext" is not available'
        ):
            create_extensions(
                connection,
                required_extensions=[extension],
            )
