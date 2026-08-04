import os
import socket
import time
import uuid
from typing import Iterator

import pytest

FALKORDB_IMAGE = "falkordb/falkordb:latest"
READINESS_TIMEOUT = 60.0


def _free_port() -> int:
    """Return a currently unused TCP port."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout: float = READINESS_TIMEOUT) -> bool:
    """Poll ``url`` until FalkorDB answers a trivial query."""
    from falkordb import FalkorDB

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            FalkorDB.from_url(url).select_graph("readiness").query("RETURN 1")
            return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def falkordb_url() -> Iterator[str]:
    """
    URL of a FalkorDB instance usable by the tests.

    Uses ``FALKORDB_TEST_URL`` when set, otherwise starts a throwaway container
    on a free port. Skips (rather than fails) when neither is available.
    """
    url = os.environ.get("FALKORDB_TEST_URL")
    if url:
        if not _wait_until_ready(url, timeout=10):
            pytest.skip(f"FalkorDB is not reachable at {url}")
        yield url
        return

    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"Docker is not available: {exc}")

    port = _free_port()
    container = client.containers.run(
        FALKORDB_IMAGE,
        detach=True,
        auto_remove=True,
        name=f"falkordb_test_{uuid.uuid4().hex[:8]}",
        ports={"6379/tcp": port},
    )
    try:
        url = f"redis://localhost:{port}"
        if not _wait_until_ready(url):  # pragma: no cover
            pytest.skip("FalkorDB container did not become ready in time")
        yield url
    finally:
        try:
            container.stop(timeout=5)
        except Exception:  # pragma: no cover
            pass


@pytest.fixture()
def graph_name() -> str:
    """A unique graph name so tests never share state."""
    return f"test_{uuid.uuid4().hex}"
