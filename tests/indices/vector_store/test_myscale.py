"""Test MyScale indexes."""

from typing import List

import pytest

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None  # type: ignore

from gpt_index.indices.vector_store import GPTMyScaleIndex
from gpt_index.readers.schema.base import Document

# local test only, update variable here for test
MYSCALE_CLUSTER_URL = None
MYSCALE_USERNAME = None
MYSCALE_CLUSTER_PASSWORD = None


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text=doc_text, extra_info={})]


@pytest.mark.skipif(
    clickhouse_connect is None
    or MYSCALE_CLUSTER_URL is None
    or MYSCALE_USERNAME is None
    or MYSCALE_CLUSTER_PASSWORD is None,
    reason="myscale-client not configured",
)
def test_build_myscale(documents: List[Document]) -> None:
    client = clickhouse_connect.get_client(
        host=MYSCALE_CLUSTER_URL,
        port=8443,
        username=MYSCALE_USERNAME,
        password=MYSCALE_CLUSTER_PASSWORD,
    )
    index = GPTMyScaleIndex.from_documents(documents, myscale_client=client)
    response = index.query("How many tests?")
    assert str(response).strip() == ("Three tests.")
