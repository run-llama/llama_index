"""Test document summary retrievers."""
from typing import List

from llama_index.indices.service_context import ServiceContext
from llama_index.schema import Document


def test_retrieve_default(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test retrieve default."""
    # TODO: add retrieve tests
