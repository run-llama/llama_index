import asyncio
from typing import Any, List

import pytest
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter


def _split_sentences(text: str) -> List[str]:
    return text.split(". ")


def _make_splitter(kind: str, include_metadata: bool) -> Any:
    if kind == "token":
        return TokenTextSplitter(
            chunk_size=20,
            chunk_overlap=0,
            tokenizer=list,
            include_metadata=include_metadata,
        )
    return SentenceSplitter(
        chunk_size=20,
        chunk_overlap=0,
        tokenizer=list,
        chunking_tokenizer_fn=_split_sentences,
        include_metadata=include_metadata,
    )


def _parse(splitter: Any, entry: str) -> List[Any]:
    documents = [Document(text="word " * 12, metadata={"context": "x" * 100})]
    if entry == "sync":
        return splitter.get_nodes_from_documents(documents)
    return asyncio.run(splitter.aget_nodes_from_documents(documents))


@pytest.mark.parametrize("kind", ["token", "sentence"])
@pytest.mark.parametrize("entry", ["sync", "async"])
def test_disabled_metadata_does_not_consume_chunk_budget(kind: str, entry: str) -> None:
    splitter = _make_splitter(kind, include_metadata=False)
    nodes = _parse(splitter, entry)

    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata == {}


@pytest.mark.parametrize("kind", ["token", "sentence"])
@pytest.mark.parametrize("entry", ["sync", "async"])
def test_enabled_metadata_still_rejects_oversized_metadata(kind: str, entry: str) -> None:
    splitter = _make_splitter(kind, include_metadata=True)

    with pytest.raises(ValueError, match="Metadata length"):
        _parse(splitter, entry)
