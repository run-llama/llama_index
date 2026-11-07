"""
Regression tests for the splitter chunk budget when ``include_metadata`` is
False (issue #23683).

``MetadataAwareTextSplitter._get_metadata_str()`` used to format a document's
metadata even when the splitter was constructed with ``include_metadata=False``,
even though the resulting nodes never contain it. Oversized metadata then
raised a ``Metadata length (...) is longer than chunk size`` error and blocked
ingestion despite the explicit opt-out. Metadata that is disabled should not
reserve any of the chunk budget; with metadata enabled, rejecting oversized
metadata remains correct.

A character tokenizer is supplied so no external tokenizer resources are
needed.
"""

import pytest
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

BIG_METADATA = {"context": "x" * 100}


def test_token_splitter_disabled_metadata_does_not_consume_budget() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = splitter.get_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )

    assert len(nodes) > 1
    assert all(node.metadata == {} for node in nodes)


@pytest.mark.asyncio
async def test_token_splitter_disabled_metadata_async_path() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = await splitter.aget_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )

    assert len(nodes) > 1
    assert all(node.metadata == {} for node in nodes)


def test_token_splitter_enabled_metadata_still_rejects_oversized() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError, match="Metadata length .* longer than chunk size"):
        splitter.get_nodes_from_documents(
            [Document(text="word " * 12, metadata=BIG_METADATA)]
        )


def test_sentence_splitter_disabled_metadata_does_not_consume_budget() -> None:
    splitter = SentenceSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = splitter.get_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )

    assert len(nodes) > 1
    assert all(node.metadata == {} for node in nodes)


def test_sentence_splitter_enabled_metadata_still_rejects_oversized() -> None:
    splitter = SentenceSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError, match="Metadata length .* longer than chunk size"):
        splitter.get_nodes_from_documents(
            [Document(text="word " * 12, metadata=BIG_METADATA)]
        )
