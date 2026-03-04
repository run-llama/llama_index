"""
Unit tests for SlideNodeParser.

Covers:
- Synchronous parsing (`get_nodes_from_documents`) under various chunk_size, window_size, and llm_workers.
- Asynchronous parsing (`aget_nodes_from_documents`) under the same parameter sets.
- Parallelism of async LLM calls (`achat`) via run_jobs, ensuring overlap when llm_workers > 1.
- Correct per-chunk invocation counts for both sync (`chat`) and async (`achat`).
- Edge‐case behavior for empty documents (no nodes returned, warning emitted).
- Handling of inputs shorter than the window (still produces valid nodes with context).
"""

import asyncio
import pytest
import warnings
from llama_index.core import Document
from llama_index.core.llms import MockLLM
from llama_index.node_parser.slide import SlideNodeParser
from unittest.mock import patch, AsyncMock


@pytest.mark.parametrize(
    ("chunk_size", "window_size", "llm_workers"),
    [
        (1, 1, 1),
        (10, 3, 2),
        (100, 5, 4),
        (390, 10, 8),
    ],
)
def test_sync_parsing_no_errors(chunk_size, window_size, llm_workers):
    """
    Integration test to ensure the sync parsing path (get_nodes_from_documents)
    completes without errors under various parameters.
    """
    # Patch the blocking LLM.chat so it returns instantly
    with patch.object(MockLLM, "chat", return_value="ctx"):
        llm = MockLLM()
        parser = SlideNodeParser.from_defaults(
            llm=llm,
            chunk_size=chunk_size,
            window_size=window_size,
            llm_workers=llm_workers,
        )

        # Three sample documents with multiple sentences
        doc1 = Document(text="This is the first document. It has some sentences.")
        doc2 = Document(text="This is the second document. Different content.")
        doc3 = Document(text="And this is the third one. More text here.")

        # Should run without raising, regardless of chunk_size/window_size/llm_workers
        nodes = parser.get_nodes_from_documents([doc1, doc2, doc3])

        # Basic sanity checks
        assert isinstance(nodes, list)
        # Every node must have the local_context metadata set
        assert all("local_context" in node.metadata for node in nodes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunk_size", "window_size", "llm_workers"),
    [(1, 1, 1), (10, 3, 2), (100, 5, 4), (390, 10, 8)],
)
async def test_async_parsing_no_errors(chunk_size, window_size, llm_workers):
    """
    Integration test to ensure the async parsing path (_aget_nodes_from_documents)
    completes without errors under various parameters.
    """
    # Patch the async LLM call so it returns immediately
    with patch.object(MockLLM, "achat", new=AsyncMock(return_value="ctx")):
        llm = MockLLM()
        parser = SlideNodeParser.from_defaults(
            llm=llm,
            chunk_size=chunk_size,
            window_size=window_size,
            llm_workers=llm_workers,
        )

        # Sample documents with multiple sentences
        doc1 = Document(text="This is the first document. It has some sentences.")
        doc2 = Document(text="This is the second document. Different content.")
        doc3 = Document(text="And this is the third one. More text here.")
        # Should run without raising, regardless of chunk_size/window_size/llm_workers
        nodes = await parser.aget_nodes_from_documents([doc1, doc2, doc3])

        # Basic sanity checks
        assert isinstance(nodes, list)
        # Every node should have the local_context metadata set
        assert all("local_context" in node.metadata for node in nodes)


@pytest.mark.asyncio
async def test_parallel_achat_calls():
    """Ensure that with max_workers>1, LLM calls overlap (are run in parallel)."""
    call_events = []

    # Fake achat that logs when it starts and ends
    async def fake_achat(self, messages):
        # Record the start of this call
        call_events.append(("start", asyncio.get_event_loop().time()))
        # Simulate a little work
        await asyncio.sleep(0.1)
        # Record the end
        call_events.append(("end", asyncio.get_event_loop().time()))
        return "ctx"

    # Patch the class method to use our fake_achat
    with patch.object(MockLLM, "achat", new=fake_achat):
        llm = MockLLM()
        parser = SlideNodeParser.from_defaults(
            llm=llm,
            chunk_size=2,  # ensure each sentence is its own chunk
            window_size=1,
            llm_workers=2,  # allow up to 2 concurrent calls
        )
        # Two-sentence doc → 2 chunks → 2 achat calls
        doc = Document(text="First. Second.")
        nodes = await parser.aget_nodes_from_documents([doc])

    # We should have exactly two nodes
    assert len(nodes) == 2

    # Now assert overlap: the second event in the log should be a "start"
    # (i.e. the second LLM call started before the first one finished)
    assert call_events[1][0] == "start", (
        "Expected the second LLM call to start before the first one ended, "
        f"but got call_events={call_events}"
    )


@pytest.mark.asyncio
async def test_async_aparse_nodes_with_mock_llm():
    """Ensure the async parser path calls achat() once per chunk and attaches contexts."""
    # Prepare a simple doc that will split into 3 chunks
    text = "Sentence one. Sentence two. Sentence three."
    document = Document(text=text)

    # Patch MockLLM.achat at the class level
    with patch.object(
        MockLLM, "achat", new=AsyncMock(return_value="dummy async context")
    ) as mock_achat:
        # Instantiate the LLM and parser as usual
        mock_llm = MockLLM()
        parser = SlideNodeParser.from_defaults(
            llm=mock_llm,
            chunk_size=3,  # one sentence → one chunk
            window_size=1,
            llm_workers=2,
        )

        # Run the async path
        nodes = await parser.aget_nodes_from_documents([document])

        # Verify achat() was called once per chunk/node
        assert mock_achat.call_count == len(nodes), (
            f"Expected achat() to be called {len(nodes)} times, got {mock_achat.call_count}"
        )
        # And that each node has our dummy context
        for node in nodes:
            assert node.metadata["local_context"] == "dummy async context"


def test_empty_doc():
    """Ensure passing empty docs returns an empty List[TextNode]."""
    warnings.warn(
        "WARNING: This test may fail if the context length of MockLLM is changed.\n"
        "Make sure chunk_size * window_size fits within MockLLM.context_window.",
        UserWarning,
    )

    llm = MockLLM()
    node_parser = SlideNodeParser.from_defaults(
        chunk_size=1300,  # setting non default values to match context length of mock LLM
        window_size=3,
        llm=llm,
    )
    nodes = node_parser.get_nodes_from_documents(documents=[Document(text="")])
    print(nodes)

    assert isinstance(nodes, list)
    assert nodes == []


def test_short_text_less_than_window():
    """Ensure parser handles short input without window overflow."""
    with patch.object(MockLLM, "chat", return_value="ctx"):
        parser = SlideNodeParser.from_defaults(
            llm=MockLLM(),
            chunk_size=780,
            window_size=5,
        )
        nodes = parser.get_nodes_from_documents([Document(text="One. Two.")])
        assert len(nodes) > 0
        assert all(node.metadata["local_context"] == "ctx" for node in nodes)


def test_llm_called_expected_times():
    """Ensure LLM.chat() is called once per chunk (class‐level patching)."""
    # Prepare a document with 4 sentences
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    document = Document(text=text)

    mock_llm = MockLLM()

    # Patch the chat method on the class (so instance is bound to it cleanly)
    with patch.object(MockLLM, "chat", return_value="dummy context") as mock_chat:
        # Force each sentence to become its own chunk:
        parser = SlideNodeParser.from_defaults(
            llm=mock_llm,
            chunk_size=3,  # small enough that each sentence splits out
            window_size=1,  # window of just the chunk itself
        )

        # Run parser
        nodes = parser.get_nodes_from_documents([document])

        # We expect one LLM.chat call per node returned
        assert mock_chat.call_count == len(nodes), (
            f"Expected chat() to be called {len(nodes)} times, "
            f"but got {mock_chat.call_count}"
        )
