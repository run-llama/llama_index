"""Regression tests for #23683.

A splitter built with ``include_metadata=False`` does not put the metadata into
the nodes, so it must not reserve chunk budget for it either. Formatting it
anyway made large metadata fail ingestion with a metadata-length error, on a
splitter that had explicitly opted out.
"""

import pytest
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter


def _document() -> Document:
    return Document(text="word " * 12, metadata={"context": "x" * 100})


def test_token_splitter_ignores_metadata_budget_when_opted_out() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    # Raised ValueError: Metadata length (...) is longer than chunk size (20).
    nodes = splitter.get_nodes_from_documents([_document()])
    assert nodes
    for node in nodes:
        assert "x" * 100 not in node.get_content()


def test_sentence_splitter_ignores_metadata_budget_when_opted_out() -> None:
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0, include_metadata=False)
    nodes = splitter.get_nodes_from_documents([_document()])
    assert nodes
    for node in nodes:
        assert "x" * 100 not in node.get_content()


def test_metadata_budget_is_still_enforced_when_included() -> None:
    """The refusal the issue describes stays for splitters that DO include it."""
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError):
        splitter.get_nodes_from_documents([_document()])