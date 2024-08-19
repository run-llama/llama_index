"""Test empty index."""

from llama_index.core.data_structs.data_structs import EmptyIndexStruct
from llama_index.core.indices.empty.base import EmptyIndex


def test_empty() -> None:
    """Test build list."""
    empty_index = EmptyIndex()
    assert isinstance(empty_index.index_struct, EmptyIndexStruct)

    retriever = empty_index.as_retriever()
    nodes = retriever.retrieve("What is?")
    assert len(nodes) == 0
