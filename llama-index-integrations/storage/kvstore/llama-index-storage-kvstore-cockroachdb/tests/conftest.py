"""Pytest fixtures: spin up a CockroachDB v25.2+ container with vector enabled.

Two paths:

1. **Local container** (default): start a fresh ``cockroachdb/cockroach`` insecure
   single-node via testcontainers, enable ``feature.vector_index.enabled``, hand
   back connection params.
2. **External instance**: set ``CRDB_TEST_URL=postgresql://user:pass@host:port/db``
   to skip the container and reuse an existing cluster (useful for CI matrices
   running against multiple CRDB versions).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Generator
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

CRDB_IMAGE = os.environ.get("CRDB_IMAGE", "cockroachdb/cockroach:latest-v25.2")


def _params_from_env() -> dict[str, Any] | None:
    url = os.environ.get("CRDB_TEST_URL")
    if not url:
        return None
    from urllib.parse import urlparse

    p = urlparse(url)
    return {
        "host": p.hostname,
        "port": p.port or 26257,
        "user": p.username or "root",
        "password": p.password,
        "database": (p.path or "/defaultdb").lstrip("/") or "defaultdb",
    }


@pytest.fixture(scope="session")
def crdb_params() -> Generator[dict[str, Any], None, None]:
    env = _params_from_env()
    if env is not None:
        yield env
        return

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = (
        DockerContainer(CRDB_IMAGE)
        .with_command("start-single-node --insecure --accept-sql-without-tls")
        .with_exposed_ports(26257, 8080)
    )
    container.start()
    try:
        wait_for_logs(container, "node startup completed", timeout=60)
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(26257))
        params = {
            "host": host,
            "port": port,
            "user": "root",
            "password": None,
            "database": "defaultdb",
        }
        _enable_vector_feature(params)
        yield params
    finally:
        container.stop()


def _enable_vector_feature(params: dict[str, Any]) -> None:
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        dbname=params["database"],
        sslmode="disable",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
    conn.close()


@pytest.fixture()
def fresh_db(crdb_params: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Create a per-test database, drop it after."""
    db_name = f"t_{uuid.uuid4().hex[:12]}"
    admin = dict(crdb_params)
    conn = psycopg2.connect(
        host=admin["host"],
        port=admin["port"],
        user=admin["user"],
        password=admin["password"],
        dbname=admin["database"],
        sslmode="disable",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE {db_name}")
    conn.close()

    test_params = dict(crdb_params)
    test_params["database"] = db_name
    yield test_params

    # Best-effort drop; tolerate races.
    for _ in range(3):
        try:
            conn = psycopg2.connect(
                host=admin["host"],
                port=admin["port"],
                user=admin["user"],
                password=admin["password"],
                dbname=admin["database"],
                sslmode="disable",
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
            conn.close()
            break
        except Exception:
            time.sleep(0.5)
