
import asyncio
import time
from typing import List, Optional

import pytest
from unittest.mock import MagicMock, AsyncMock

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import (
    IndexNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)

class MockSlowRetriever(BaseRetriever):
    """A mock retriever that takes time to respond."""
    def __init__(self, delay: float = 0.5):
        super().__init__()
        self.delay = delay

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        time.sleep(self.delay)
        return [NodeWithScore(node=TextNode(text="sync"), score=1.0)]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        await asyncio.sleep(self.delay)
        return [NodeWithScore(node=TextNode(text="async"), score=1.0)]

@pytest.mark.asyncio
async def test_base_retriever_concurrency() -> None:
    """Test that recursive retrieval runs concurrently."""
    delay = 0.5
    num_sub_retrievers = 3
    
    # Create sub-retrievers
    sub_retrievers = [MockSlowRetriever(delay=delay) for _ in range(num_sub_retrievers)]
    
    # Create IndexNodes pointing to these retrievers
    nodes = []
    for i, retriever in enumerate(sub_retrievers):
        node = IndexNode(
            text=f"link-{i}",
            index_id=f"idx-{i}",
            obj=retriever
        )
        nodes.append(NodeWithScore(node=node))

    # Parent retriever that will process these nodes
    parent_retriever = MockSlowRetriever(delay=0.0) # Parent itself is fast
    
    # Measure time for _ahandle_recursive_retrieval
    start_time = time.perf_counter()
    results = await parent_retriever._ahandle_recursive_retrieval(
        QueryBundle("test"), nodes
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Verification:
    # If sequential: duration ~= num_sub_retrievers * delay (3 * 0.5 = 1.5s)
    # If concurrent: duration ~= delay + overhead (~0.5s + small overhead)
    
    assert len(results) == num_sub_retrievers
    
    # We want to assert it was faster than sequential execution
    # Allow some buffer, but it should definitely be less than (num * delay)
    assert duration < (num_sub_retrievers * delay), f"Execution took {duration}s, expected concurrent speedup"
    assert duration >= delay, "Execution cannot be faster than a single delay"
    
    print(f"Concurrent retrieval took {duration:.4f}s (Sequential would be ~{num_sub_retrievers*delay}s)")
