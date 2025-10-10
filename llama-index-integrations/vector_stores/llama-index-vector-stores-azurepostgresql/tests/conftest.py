"""pytest fixtures for PostgreSQL connections and connection pools.

This file contains pytest fixtures for setting up PostgreSQL connections
and connection pools, both synchronous and asynchronous. It supports
basic authentication and Azure AD authentication, allowing for flexible
testing configurations.
"""

import logging
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from azure.core.credentials import TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from psycopg import AsyncConnection, Connection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool, PoolTimeout

from llama_index.vector_stores.azure_postgres.common import (
    AsyncAzurePGConnectionPool,
    AsyncConnectionInfo,
    AzurePGConnectionPool,
    BasicAuth,
    ConnectionInfo,
    SSLMode,
)
from llama_index.vector_stores.azure_postgres.common._shared import (
    TOKEN_CREDENTIAL_SCOPE,
)

logger = logging.getLogger(__name__)
postgres_not_available = False


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest command line options for PostgreSQL connection information.

    This function allows users to specify PostgreSQL connection parameters
    such as application name, database name, host, password, port, and user
    through command line options or environment variables. The defaults are
    set to common PostgreSQL defaults or environment variables if available.

    :param parser: The pytest parser object.
    :type parser: pytest.Parser
    """
    parser.addoption(
        "--pg-appname",
        action="store",
        type=str,
        default=os.getenv("PGAPPNAME", "pytest"),
        help="PostgreSQL application name (env: PGAPPNAME, default: pytest)",
    )
    parser.addoption(
        "--pg-database",
        action="store",
        type=str,
        default=os.getenv("PGDATABASE", "postgres"),
        help="PostgreSQL database name (env: PGDATABASE, default: postgres)",
    )
    parser.addoption(
        "--pg-host",
        action="store",
        type=str,
        default=os.getenv("PGHOST", "localhost"),
        help="PostgreSQL host (env: PGHOST, default: localhost)",
    )
    parser.addoption(
        "--pg-password",
        action="store",
        type=str,
        default=os.getenv("PGPASSWORD", ""),
        help="PostgreSQL password (env: PGPASSWORD, default: <empty string>)",
    )
    parser.addoption(
        "--pg-port",
        action="store",
        type=int,
        default=os.getenv("PGPORT", 5432),
        help="PostgreSQL port (env: PGPORT, default: 5432)",
    )
    parser.addoption(
        "--pg-user",
        action="store",
        type=str,
        default=os.getenv("PGUSER", "postgres"),
        help="PostgreSQL user (env: PGUSER, default: postgres)",
    )


@pytest.fixture
async def async_connection(
    async_connection_pool: AsyncConnectionPool,
) -> AsyncGenerator[AsyncConnection, Any]:
    """Fixture to provide an asynchronous PostgreSQL connection.

    :param async_connection_pool: The asynchronous connection pool (fixture) to use.
    :type async_connection_pool: AsyncConnectionPool
    :return: An asynchronous PostgreSQL connection.
    :rtype: AsyncConnection
    """
    global postgres_not_available
    if postgres_not_available:
        pytest.skip("Could not reach the database")
    try:
        async with async_connection_pool.connection() as conn:
            yield conn
    except PoolTimeout as exc:
        logger.warning("PoolTimeout %s", exc)
        postgres_not_available = True
        pytest.skip("Could not reach the database")


@pytest.fixture(scope="session")
async def async_connection_info(
    async_credentials: BasicAuth | AsyncTokenCredential,
    pytestconfig: pytest.Config,
) -> AsyncConnectionInfo:
    """Fixture to provide asynchronous connection information for PostgreSQL.

    :param async_credentials: The asynchronous credentials (fixture) to use for authentication.
    :type async_credentials: BasicAuth | AsyncTokenCredential
    :param pytestconfig: The pytest configuration object.
    :type pytestconfig: pytest.Config
    :return: An asynchronous connection information object.
    :rtype: AsyncConnectionInfo
    """
    return AsyncConnectionInfo(
        application_name=pytestconfig.getoption("pg_appname"),
        host=pytestconfig.getoption("pg_host"),
        dbname=pytestconfig.getoption("pg_database"),
        port=pytestconfig.getoption("pg_port"),
        sslmode=SSLMode.prefer,
        credentials=async_credentials,
    )


@pytest.fixture(scope="session")
async def async_connection_pool(
    async_connection_info: AsyncConnectionInfo,
) -> AsyncGenerator[AsyncConnectionPool, Any]:
    """Fixture to provide an asynchronous PostgreSQL connection pool.

    :param async_connection_info: The asynchronous connection information (fixture) to use.
    :type async_connection_info: AsyncConnectionInfo
    :return: An asynchronous PostgreSQL connection pool.
    :rtype: AsyncConnectionPool
    """

    # disable prepared statements during testing
    async def disable_prepared_statements(async_conn: AsyncConnection) -> None:
        async_conn.prepare_threshold = None

    credentials, host = async_connection_info.credentials, async_connection_info.host
    assert host is not None, "Host must be provided for connection pool"
    if isinstance(credentials, AsyncTokenCredential) and host.find("azure.com") == -1:
        pytest.skip(
            reason="Azure AD authentication requires an Azure PostgreSQL instance"
        )
    async with AsyncAzurePGConnectionPool(
        azure_conn_info=async_connection_info, configure=disable_prepared_statements
    ) as pool:
        yield pool


@pytest.fixture(scope="session", params=["azure-ad", "basic-auth"])
async def async_credentials(
    pytestconfig: pytest.Config, request: pytest.FixtureRequest
) -> BasicAuth | AsyncTokenCredential:
    """Fixture to provide asynchronous credentials for PostgreSQL.

    This fixture supports both Azure AD authentication ("azure-ad" in `request.param`)
    and basic authentication ("basic-auth" in `request.param`). When/if Azure AD
    authentication is requested, it uses the `AsyncDefaultAzureCredential` to obtain
    a token. For basic authentication, it retrieves the username and password from
    the pytest configuration options.

    When/if Azure AD authentication is not available, it skips the test with a reason.

    :param pytestconfig: The pytest configuration object.
    :type pytestconfig: pytest.Config
    :param request: The pytest fixture request object.
    :type request: pytest.FixtureRequest
    :raises ValueError: If the authentication type is unknown.
    :return: The asynchronous credentials for PostgreSQL.
    :rtype: BasicAuth | AsyncTokenCredential
    """
    if request.param == "azure-ad":
        try:
            async_credentials = AsyncDefaultAzureCredential()
            _token = await async_credentials.get_token(TOKEN_CREDENTIAL_SCOPE)
            return async_credentials
        except Exception:
            pytest.skip(reason="Azure AD authentication not available")
    elif request.param == "basic-auth":
        username = pytestconfig.getoption("pg_user")
        password = pytestconfig.getoption("pg_password")
        return BasicAuth(username=username, password=password)
    else:
        raise ValueError(f"Unknown auth type: {request.param}")


@pytest.fixture(scope="session")
async def async_schema(
    async_connection_pool: AsyncConnectionPool,
) -> AsyncGenerator[str, Any]:
    """Fixture to create and drop a schema for testing purposes.

    :param async_connection_pool: The asynchronous connection pool (fixture) to use.
    :type async_connection_pool: AsyncConnectionPool
    :return: The name of the created schema.
    :rtype: str
    """
    global postgres_not_available
    if postgres_not_available:
        pytest.skip("Could not reach the database")

    try:
        async with (
            async_connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                sql.SQL(
                    """
                    select  oid as schema_id, nspname as schema_name
                    from  pg_namespace
                    """
                )
            )
            resultset = await cursor.fetchall()
            schema_names = [row["schema_name"] for row in resultset]
    except PoolTimeout as exc:
        logger.warning("PoolTimeout %s", exc)
        postgres_not_available = True
        pytest.skip("Could not reach the database")

    _schema: str | None = None
    for idx in range(100_000):
        _schema_name = f"pytest-{idx:05d}"
        if _schema_name not in schema_names:
            _schema = _schema_name
            break
    if _schema is None:
        pytest.fail("Could not find a unique schema name for testing")

    async with async_connection_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            sql.SQL("create schema {schema}").format(schema=sql.Identifier(_schema))
        )

    yield _schema

    async with async_connection_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            sql.SQL("drop schema {schema} cascade").format(
                schema=sql.Identifier(_schema)
            )
        )


@pytest.fixture
def connection(connection_pool: ConnectionPool) -> Generator[Connection, Any, None]:
    """Fixture to provide a PostgreSQL connection.

    :param connection_pool: The connection pool (fixture) to use.
    :type connection_pool: ConnectionPool
    :return: A PostgreSQL connection.
    :rtype: Connection
    """
    global postgres_not_available
    if postgres_not_available:
        pytest.skip("Could not reach the database")
    try:
        with connection_pool.connection() as conn:
            yield conn
    except PoolTimeout as exc:
        logger.warning("PoolTimeout %s", exc)
        postgres_not_available = True
        pytest.skip("Could not reach the database")


@pytest.fixture(scope="session")
def connection_info(
    credentials: BasicAuth | TokenCredential,
    pytestconfig: pytest.Config,
) -> ConnectionInfo:
    """Fixture to provide connection information for PostgreSQL.

    :param credentials: The credentials (fixture) to use for authentication.
    :type credentials: BasicAuth | TokenCredential
    :param pytestconfig: The pytest configuration object.
    :type pytestconfig: pytest.Config
    :return: The connection information for PostgreSQL.
    :rtype: ConnectionInfo
    """
    return ConnectionInfo(
        application_name=pytestconfig.getoption("pg_appname"),
        host=pytestconfig.getoption("pg_host"),
        dbname=pytestconfig.getoption("pg_database"),
        port=pytestconfig.getoption("pg_port"),
        sslmode=SSLMode.prefer,
        credentials=credentials,
    )


@pytest.fixture(scope="session")
def connection_pool(
    connection_info: ConnectionInfo,
) -> Generator[ConnectionPool, Any, None]:
    """Fixture to provide a PostgreSQL connection pool.

    :param connection_info: The connection information (fixture) to use.
    :type connection_info: ConnectionInfo
    :return: A PostgreSQL connection pool.
    :rtype: ConnectionPool
    """

    # disable prepared statements during testing (needed for failures in (a)add_texts)
    def disable_prepared_statements(conn: Connection) -> None:
        conn.prepare_threshold = None

    credentials, host = connection_info.credentials, connection_info.host
    assert host is not None, "Host must be provided for connection pool"
    if isinstance(credentials, TokenCredential) and host.find("azure.com") == -1:
        pytest.skip(
            reason="Azure AD authentication requires an Azure PostgreSQL instance"
        )
    with AzurePGConnectionPool(
        azure_conn_info=connection_info, configure=disable_prepared_statements
    ) as pool:
        yield pool


@pytest.fixture(scope="session", params=["azure-ad", "basic-auth"])
def credentials(
    pytestconfig: pytest.Config, request: pytest.FixtureRequest
) -> BasicAuth | TokenCredential:
    """Fixture to provide credentials for PostgreSQL.

    This fixture supports both Azure AD authentication ("azure-ad" in `request.param`)
    and basic authentication ("basic-auth" in `request.param`). When/if Azure AD
    authentication is requested, it uses the `DefaultAzureCredential` to obtain
    a token. For basic authentication, it retrieves the username and password from
    the pytest configuration options.

    When/if Azure AD authentication is not available, it skips the test with a reason.

    :param pytestconfig: The pytest configuration object.
    :type pytestconfig: pytest.Config
    :param request: The pytest fixture request object.
    :type request: pytest.FixtureRequest
    :raises ValueError: If the authentication type is unknown.
    :return: The credentials for PostgreSQL.
    :rtype: BasicAuth | TokenCredential
    """
    if request.param == "azure-ad":
        try:
            credentials = DefaultAzureCredential()
            _token = credentials.get_token(TOKEN_CREDENTIAL_SCOPE)
            return credentials
        except Exception:
            pytest.skip(reason="Azure AD authentication not available")
    elif request.param == "basic-auth":
        username = pytestconfig.getoption("pg_user")
        password = pytestconfig.getoption("pg_password")
        return BasicAuth(username=username, password=password)
    else:
        raise ValueError(f"Unknown auth type: {request.param}")


@pytest.fixture(scope="session")
def schema(connection_pool: ConnectionPool) -> Generator[str, Any, None]:
    """Fixture to create and drop a schema for testing purposes.

    :param connection_pool: The connection pool (fixture) to use.
    :type connection_pool: ConnectionPool
    :return: The name of the created schema.
    :rtype: str
    """
    global postgres_not_available
    if postgres_not_available:
        pytest.skip("Could not reach the database")

    try:
        with (
            connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
                    """
                    select oid as schema_id, nspname as schema_name
                    from pg_namespace
                    """
                )
            )
            resultset = cursor.fetchall()
            schema_names = [row["schema_name"] for row in resultset]
    except PoolTimeout as exc:
        logger.warning("PoolTimeout %s", exc)
        postgres_not_available = True
        pytest.skip("Could not reach the database")

    _schema: str | None = None
    for idx in range(100_000):
        _schema_name = f"pytest-{idx:05d}"
        if _schema_name not in schema_names:
            _schema = _schema_name
            break
    if _schema is None:
        pytest.fail("Could not find a unique schema name for testing")

    with connection_pool.connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            sql.SQL("create schema {schema}").format(schema=sql.Identifier(_schema))
        )

    yield _schema

    with connection_pool.connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            sql.SQL("drop schema {schema} cascade").format(
                schema=sql.Identifier(_schema)
            )
        )
