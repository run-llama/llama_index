"""Test empty index."""

from llama_index.data_structs.data_structs import EmptyIndex
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.service_context import ServiceContext


def test_empty(
    mock_service_context: ServiceContext,
) -> None:
    """Test build list."""
    empty_index = EmptyIndex(service_context=mock_service_context)
    assert isinstance(empty_index.index_struct, EmptyIndex)

    retriever = empty_index.as_retriever()
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 0
