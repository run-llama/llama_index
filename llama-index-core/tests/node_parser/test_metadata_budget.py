from typing import Any, Dict, List

import pytest

from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode


@pytest.mark.asyncio
@pytest.mark.parametrize("splitter_class", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_disabled_metadata_does_not_consume_chunk_budget(
    splitter_class: type,
    use_async: bool,
    include_metadata: bool,
) -> None:
    options: Dict[str, Any] = {
        "chunk_size": 20,
        "chunk_overlap": 0,
        "tokenizer": list,
        "include_metadata": include_metadata,
    }
    if splitter_class is SentenceSplitter:
        options["chunking_tokenizer_fn"] = lambda text: [text]
    splitter = splitter_class(**options)
    document = Document(text="word " * 12, metadata={"context": "x" * 100})

    async def parse() -> List[BaseNode]:
        if use_async:
            return await splitter.aget_nodes_from_documents([document])
        return splitter.get_nodes_from_documents([document])

    if include_metadata:
        with pytest.raises(ValueError, match="Metadata length"):
            await parse()
        return

    nodes = await parse()

    assert len(nodes) > 1
    assert "".join(node.text.replace(" ", "") for node in nodes) == "word" * 12
    for node in nodes:
        assert node.metadata == {}
        assert len(node.text) <= 20
        assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.text
        assert node.source_node is not None
        assert node.source_node.node_id == document.id_


def _word_splitter(
    splitter_class: type, chunk_size: int, include_metadata: bool = True
) -> MetadataAwareTextSplitter:
    options: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": 0,
        "tokenizer": str.split,
        "include_metadata": include_metadata,
    }
    if splitter_class is SentenceSplitter:
        options["chunking_tokenizer_fn"] = lambda text: [text]
    return splitter_class(**options)


@pytest.mark.parametrize("splitter_class", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_empty_metadata_uses_full_budget_without_warning(
    splitter_class: type, chunk_size: int, capsys: pytest.CaptureFixture[str]
) -> None:
    splitter = _word_splitter(splitter_class, chunk_size)
    words = ["one", "two", "three", "four", "five", "six", "seven"]

    chunks = splitter.split_text_metadata_aware(" ".join(words), metadata_str="")

    assert chunks == [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("splitter_class", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("use_async", [False, True])
async def test_disabled_metadata_uses_full_budget_without_warning(
    splitter_class: type, use_async: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    splitter = _word_splitter(splitter_class, chunk_size=5, include_metadata=False)
    document = Document(
        text="one two three four five six seven", metadata={"context": "x" * 100}
    )

    nodes = (
        await splitter.aget_nodes_from_documents([document])
        if use_async
        else splitter.get_nodes_from_documents([document])
    )

    assert [node.get_content() for node in nodes] == [
        "one two three four five",
        "six seven",
    ]
    assert all(node.metadata == {} for node in nodes)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("splitter_class", "metadata_budget"),
    [(SentenceSplitter, 2), (TokenTextSplitter, 4)],
)
@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("effective_chunk_size", [-1, 0, 1, 49, 50])
def test_explicit_metadata_preserves_budget_and_warnings(
    splitter_class: type,
    metadata_budget: int,
    include_metadata: bool,
    effective_chunk_size: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunk_size = metadata_budget + effective_chunk_size
    splitter = _word_splitter(splitter_class, chunk_size, include_metadata)
    text = " ".join(["word"] * (max(effective_chunk_size, 0) + 1))

    if effective_chunk_size <= 0:
        with pytest.raises(ValueError, match="Metadata length.*longer than chunk size"):
            splitter.split_text_metadata_aware(text, metadata_str="category science")
        assert capsys.readouterr().out == ""
        return

    chunks = splitter.split_text_metadata_aware(text, metadata_str="category science")

    assert chunks == [" ".join(["word"] * effective_chunk_size), "word"]
    output = capsys.readouterr().out
    if effective_chunk_size < 50:
        assert f"Metadata length ({metadata_budget}) is close to chunk size" in output
        assert "Resulting chunks are less than 50 tokens." in output
    else:
        assert output == ""
