"""Test document summary retrievers."""
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.schema.base import Document

from typing import List


def test_retrieve_default(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test retrieve default."""
    # TODO: add retrieve tests
    pass
