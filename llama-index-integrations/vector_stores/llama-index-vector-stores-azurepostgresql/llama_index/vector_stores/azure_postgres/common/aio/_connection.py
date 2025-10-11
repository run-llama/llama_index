"""Asynchronous connection handling for Azure Database for PostgreSQL."""

import logging
import time
from collections.abc import Awaitable, Callable

from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import ConfigDict

from .._shared import (
    TOKEN_CREDENTIAL_SCOPE,
    BaseConnectionInfo,
    BasicAuth,
    Extension,
    get_username_password,
    run_coroutine_in_sync,
)

_logger = logging.getLogger(__name__)


async def async_check_connection(
    conn: AsyncConnection, /, required_extensions: list[Extension] = []
):
    """Check if the connection to Azure Database for PostgreSQL is valid and required extensions are installed.

    :param conn: Async connection to the Azure Database for PostgreSQL.
    :type conn: AsyncConnection
    :param required_extensions: List of required extensions to check if they are installed.
    :type required_extensions: list[Extension]
    :raises RuntimeError: If the connection check fails or required extensions are not installed.
    """
    async with conn.cursor(row_factory=dict_row) as cursor:
        _logger.debug("checking connection")
        t_start = time.perf_counter()
        await cursor.execute("select 1")
        result = await cursor.fetchone()
        t_elapsed = time.perf_counter() - t_start
        assert result is not None, "Connection check failed: no result returned."
        _logger.debug(
            "connection check successful. elapsed time: %.3f ms", t_elapsed * 1000
        )

        for ext in required_extensions:
            ext_name = ext.ext_name
            ext_version = ext.ext_version
            schema_name = ext.schema_name
            await cursor.execute(
                sql.SQL(
                    """
                    select  extname as ext_name, extversion as ext_version,
                            n.nspname as schema_name
                      from  pg_extension e
                            left join pg_namespace n on e.extnamespace = n.oid
                     where  extname = %(ext_name)s
                    """
                ),
                {"ext_name": ext_name},
            )
            resultset = await cursor.fetchone()
            if resultset is None:
                raise RuntimeError(f"Required extension '{ext_name}' is not installed.")
            if ext_version is not None and resultset["ext_version"] != ext_version:
                raise RuntimeError(
                    f"Required extension '{ext_name}' version mismatch: "
                    f"expected {ext_version}, got {resultset['ext_version']}."
                )
            if schema_name is not None and resultset["schema_name"] != schema_name:
                raise RuntimeError(
                    f"Required extension '{ext_name}' is not installed in the expected schema: "
                    f"expected {schema_name}, got {resultset['schema_name']}."
                )
            _logger.debug(
                "required extension '%s' is installed (version: %s, schema: %s)",
                resultset["ext_name"],
                resultset["ext_version"],
                resultset["schema_name"],
            )


async def async_create_extensions(
    conn: AsyncConnection, /, required_extensions: list[Extension] = []
):
    """Create required extensions in the Azure Database for PostgreSQL connection.

    :param conn: Async connection to the Azure Database for PostgreSQL.
    :type conn: AsyncConnection
    :param required_extensions: List of required extensions to create.
    :type required_extensions: list[Extension]
    :raises Exception: If the connection is not valid or if an error occurs during extension creation.
    """
    async with conn.cursor() as cursor:
        for ext in required_extensions:
            ext_name = ext.ext_name
            ext_version = ext.ext_version
            schema_name = ext.schema_name
            cascade = ext.cascade
            _logger.debug(
                "creating extension (if not exists): %s (version: %s, schema: %s, cascade: %s)",
                ext_name,
                ext_version,
                schema_name,
                cascade,
            )
            await cursor.execute(
                sql.SQL(
                    """
                    create extension  if not exists {ext_name}
                                with  {schema_expr}
                                      {version_expr}
                                      {cascade_expr}
                    """
                ).format(
                    ext_name=sql.Identifier(ext_name),
                    schema_expr=sql.SQL("schema {schema_name}").format(
                        schema_name=sql.Identifier(schema_name)
                    )
                    if schema_name is not None
                    else sql.SQL(""),
                    version_expr=sql.SQL("version {version}").format(
                        version=sql.Literal(ext_version)
                    )
                    if ext_version is not None
                    else sql.SQL(""),
                    cascade_expr=sql.SQL("cascade") if cascade else sql.SQL(""),
                )
            )


class AsyncConnectionInfo(BaseConnectionInfo):
    """Base connection information for Azure Database for PostgreSQL connections.

    :param host: Hostname of the Azure Database for PostgreSQL server.
    :type host: str | None
    :param dbname: Name of the database to connect to.
    :type dbname: str
    :param port: Port number for the connection.
    :type port: int
    :param credentials: Credentials for authentication.
    :type credentials: BasicAuth | AsyncTokenCredential
    :param sslmode: SSL mode for the connection.
    :type sslmode: SSLMode
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # True to allow AsyncTokenCredential
    )

    credentials: BasicAuth | AsyncTokenCredential = DefaultAzureCredential()


class AsyncAzurePGConnectionPool(AsyncConnectionPool):
    """Async connection pool for Azure Database for PostgreSQL connections."""

    def __init__(
        self,
        conninfo: str = "",
        *,
        azure_conn_info: AsyncConnectionInfo = AsyncConnectionInfo(),
        **kwargs,
    ):
        if isinstance(azure_conn_info.credentials, AsyncTokenCredential):
            credential_provider = azure_conn_info.credentials
            coroutine = credential_provider.get_token(TOKEN_CREDENTIAL_SCOPE)

            _logger.debug(
                "getting token from TokenCredential for the scope: %s",
                TOKEN_CREDENTIAL_SCOPE,
            )

            token = run_coroutine_in_sync(coroutine)

            _logger.info("getting username and password from token")
            username, password = get_username_password(token)

            _logger.debug("wrapping reconnect_failed function")
            reconnect_failed: (
                Callable[[AsyncConnectionPool], Awaitable[None]] | None
            ) = kwargs.get("reconnect_failed")

            async def reconnect_failed_wrapper(pool: AsyncConnectionPool) -> None:
                if reconnect_failed:
                    await reconnect_failed(pool)

                _logger.debug(
                    "getting token from TokenCredential for the scope: %s",
                    TOKEN_CREDENTIAL_SCOPE,
                )
                token = await credential_provider.get_token(TOKEN_CREDENTIAL_SCOPE)

                _logger.info("getting username and password from token")
                username, password = get_username_password(token)

                pool.kwargs.update(
                    user=username,
                    password=password,
                )

            kwargs["reconnect_failed"] = reconnect_failed_wrapper
        else:
            username, password = get_username_password(azure_conn_info.credentials)

        azure_conn_info_kwargs = azure_conn_info.model_dump(
            mode="json", exclude_none=True, exclude=set(["credentials"])
        )
        _logger.debug(
            "updating AsyncConnectionPool kwargs with those from: %s",
            azure_conn_info_kwargs,
        )
        kwargs_ = kwargs.get("kwargs", {})
        kwargs_.update(user=username, password=password, **azure_conn_info_kwargs)
        kwargs["kwargs"] = kwargs_

        super().__init__(conninfo, **kwargs)
