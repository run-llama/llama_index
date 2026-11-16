from typing import List, Type

import pytest
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import Document, MetadataMode


def _byte_tokenizer(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def _document(dense_mode: MetadataMode) -> Document:
    return Document(
        text="word " * 160,
        metadata={"ascii": "a" * 120, "cjk": "界" * 60},
        excluded_embed_metadata_keys=[
            "ascii" if dense_mode == MetadataMode.EMBED else "cjk"
        ],
        excluded_llm_metadata_keys=[
            "cjk" if dense_mode == MetadataMode.EMBED else "ascii"
        ],
        text_template="{metadata_str}{content}",
    )


@pytest.mark.parametrize("splitter_cls", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("dense_mode", [MetadataMode.EMBED, MetadataMode.LLM])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_metadata_budget_uses_token_length(
    splitter_cls: Type[SentenceSplitter | TokenTextSplitter],
    dense_mode: MetadataMode,
    use_async: bool,
) -> None:
    # Shorter CJK metadata costs more byte tokens than the longer ASCII metadata.
    document = _document(dense_mode)
    kwargs = (
        {"chunking_tokenizer_fn": lambda text: [text]}
        if splitter_cls is SentenceSplitter
        else {}
    )
    splitter = splitter_cls(
        chunk_size=256,
        chunk_overlap=0,
        tokenizer=_byte_tokenizer,
        **kwargs,
    )

    nodes = (
        await splitter.aget_nodes_from_documents([document])
        if use_async
        else splitter.get_nodes_from_documents([document])
    )

    assert nodes
    assert " ".join(node.text for node in nodes) == document.text.strip()
    for node in nodes:
        assert node.metadata == document.metadata
        assert node.ref_doc_id == document.id_
        for mode in (MetadataMode.EMBED, MetadataMode.LLM):
            assert len(_byte_tokenizer(node.get_content(metadata_mode=mode))) <= 256


@pytest.mark.parametrize("splitter_cls", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("dense_mode", [MetadataMode.EMBED, MetadataMode.LLM])
def test_metadata_over_token_budget_is_rejected(
    splitter_cls: Type[SentenceSplitter | TokenTextSplitter],
    dense_mode: MetadataMode,
) -> None:
    kwargs = (
        {"chunking_tokenizer_fn": lambda text: [text]}
        if splitter_cls is SentenceSplitter
        else {}
    )
    splitter = splitter_cls(
        chunk_size=160,
        chunk_overlap=0,
        tokenizer=_byte_tokenizer,
        **kwargs,
    )

    with pytest.raises(
        ValueError, match="Metadata length .* is longer than chunk size"
    ):
        splitter.get_nodes_from_documents([_document(dense_mode)])


@pytest.mark.parametrize("splitter_cls", [SentenceSplitter, TokenTextSplitter])
@pytest.mark.parametrize("metadata", [{}, {"title": "short title"}])
def test_metadata_budget_preserves_normal_chunks(
    splitter_cls: Type[SentenceSplitter | TokenTextSplitter], metadata: dict
) -> None:
    kwargs = (
        {"chunking_tokenizer_fn": lambda text: [text]}
        if splitter_cls is SentenceSplitter
        else {}
    )
    splitter = splitter_cls(
        chunk_size=256,
        chunk_overlap=0,
        tokenizer=_byte_tokenizer,
        **kwargs,
    )
    document = Document(text="short document", metadata=metadata)

    nodes = splitter.get_nodes_from_documents([document])

    assert len(nodes) == 1
    assert nodes[0].text == document.text
    assert nodes[0].metadata == document.metadata
