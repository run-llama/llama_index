"""
Regression tests: disabled metadata must not consume the splitter budget.

Issue #23683: SentenceSplitter and TokenTextSplitter reserved space for a
document's metadata even when constructed with include_metadata=False, so large
metadata could fail ingestion with a metadata-length error despite the explicit
opt-out. The fix makes MetadataAwareTextSplitter._get_metadata_str return an
empty string when include_metadata is False.

These tests use a character tokenizer (tokenizer=list) and need no API calls or
downloaded resources, so they run deterministically in any environment.
"""

import pytest

from llama_index.core.node_parser.text import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import Document


def _oversized_metadata_doc() -> Document:
    # 12 short words of text but a 100-char metadata field. With a char
    # tokenizer the metadata alone is far larger than the 20-token budget.
    return Document(text="word " * 12, metadata={"context": "x" * 100})


def test_token_splitter_disabled_metadata_no_error() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = splitter.get_nodes_from_documents([_oversized_metadata_doc()])
    assert len(nodes) >= 1
    for node in nodes:
        assert node.metadata == {}


@pytest.mark.asyncio
async def test_token_splitter_disabled_metadata_no_error_async() -> None:
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = await splitter.aget_nodes_from_documents([_oversized_metadata_doc()])
    assert len(nodes) >= 1
    for node in nodes:
        assert node.metadata == {}


def test_sentence_splitter_disabled_metadata_no_error() -> None:
    splitter = SentenceSplitter(
        chunk_size=20,
        chunk_overlap=0,
        tokenizer=list,
        chunking_tokenizer_fn=lambda text: [text],
        include_metadata=False,
    )
    nodes = splitter.get_nodes_from_documents([_oversized_metadata_doc()])
    assert len(nodes) >= 1
    for node in nodes:
        assert node.metadata == {}


@pytest.mark.asyncio
async def test_sentence_splitter_disabled_metadata_no_error_async() -> None:
    splitter = SentenceSplitter(
        chunk_size=20,
        chunk_overlap=0,
        tokenizer=list,
        chunking_tokenizer_fn=lambda text: [text],
        include_metadata=False,
    )
    nodes = await splitter.aget_nodes_from_documents([_oversized_metadata_doc()])
    assert len(nodes) >= 1
    for node in nodes:
        assert node.metadata == {}


def test_token_splitter_enabled_oversized_metadata_still_errors() -> None:
    # Control: with metadata enabled, an oversized metadata field must still be
    # rejected (the opt-in behavior is unchanged).
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError):
        splitter.get_nodes_from_documents([_oversized_metadata_doc()])


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 20])
def test_disabled_metadata_uses_full_token_budget(chunk_size: int) -> None:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        tokenizer=list,
        include_metadata=False,
    )
    document = Document(text="abcdefghij", metadata={"context": "x" * 100})
    nodes = splitter.get_nodes_from_documents([document])
    assert [node.text for node in nodes] == splitter.split_text(document.text)
    assert all(node.metadata == {} for node in nodes)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 20])
async def test_disabled_metadata_uses_full_token_budget_async(chunk_size: int) -> None:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        tokenizer=list,
        include_metadata=False,
    )
    document = Document(text="abcdefghij", metadata={"context": "x" * 100})
    nodes = await splitter.aget_nodes_from_documents([document])
    assert [node.text for node in nodes] == splitter.split_text(document.text)
    assert all(node.metadata == {} for node in nodes)


def test_enabled_metadata_retains_format_budget() -> None:
    splitter = TokenTextSplitter(
        chunk_size=2, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError, match="Metadata length"):
        splitter.get_nodes_from_documents([Document(text="abc")])
