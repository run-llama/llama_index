"""Regression tests: disabled metadata must not consume the splitter budget.

Issue #23683: SentenceSplitter and TokenTextSplitter reserved space for a
document's metadata even when constructed with include_metadata=False, so large
metadata could fail ingestion with a metadata-length error despite the explicit
opt-out. The fix makes MetadataAwareTextSplitter._get_metadata_str return an
empty string when include_metadata is False.

These tests use a character tokenizer (tokenizer=list) and need no API calls or
downloaded resources, so they run deterministically in any environment.
"""
import nltk
import pytest

from llama_index.core.node_parser.text import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import Document


def _nltk_punkt_available() -> bool:
    try:
        nltk.data.find("tokenizers/punkt")
        return True
    except LookupError:
        return False


# SentenceSplitter relies on NLTK "punkt" for sentence tokenization; skip when
# that corpus is unavailable so the suite still runs in minimal environments.
_skip_no_punkt = pytest.mark.skipif(
    not _nltk_punkt_available(),
    reason="NLTK 'punkt' data not available in this environment",
)


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


@_skip_no_punkt
def test_sentence_splitter_disabled_metadata_no_error() -> None:
    splitter = SentenceSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )
    nodes = splitter.get_nodes_from_documents([_oversized_metadata_doc()])
    assert len(nodes) >= 1
    for node in nodes:
        assert node.metadata == {}


@_skip_no_punkt
@pytest.mark.asyncio
async def test_sentence_splitter_disabled_metadata_no_error_async() -> None:
    splitter = SentenceSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
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
