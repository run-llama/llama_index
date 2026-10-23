from typing import Any, Dict, List

import pytest

from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
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
