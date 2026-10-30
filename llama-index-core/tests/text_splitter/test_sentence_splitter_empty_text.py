from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document


def _splitter() -> SentenceSplitter:
    """Splitter with no NLTK/tiktoken dependency (char tokenizer, no-op sentences)."""
    return SentenceSplitter(
        chunk_size=100,
        chunk_overlap=0,
        tokenizer=list,
        chunking_tokenizer_fn=lambda t: [t],
    )


def test_split_text_empty_returns_no_chunks() -> None:
    """Empty text yields no chunks, consistent with whitespace-only text."""
    splitter = _splitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []
    assert splitter.split_text("\n\n") == []
    # non-empty text is unaffected
    assert splitter.split_text("hello world") == ["hello world"]


def test_split_text_metadata_aware_empty_returns_no_chunks() -> None:
    """The metadata-aware path behaves the same for empty text."""
    splitter = _splitter()
    assert splitter.split_text_metadata_aware("", metadata_str="") == []
    assert splitter.split_text_metadata_aware("   ", metadata_str="") == []
    assert splitter.split_text_metadata_aware("hello world", metadata_str="") == [
        "hello world"
    ]


def test_empty_document_produces_no_nodes() -> None:
    """An empty Document must not become a blank node in the index."""
    splitter = _splitter()
    assert splitter.get_nodes_from_documents([Document(text="")]) == []

    nodes = splitter.get_nodes_from_documents([Document(text="hello world")])
    assert len(nodes) == 1
    assert nodes[0].text == "hello world"


def test_mixed_batch_skips_only_empty_document() -> None:
    """In a mixed batch, the empty document is skipped and the rest are kept."""
    splitter = _splitter()
    nodes = splitter.get_nodes_from_documents(
        [Document(text=""), Document(text="first doc"), Document(text="   ")]
    )
    assert [node.text for node in nodes] == ["first doc"]
