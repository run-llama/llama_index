import base64
from unittest.mock import MagicMock, patch


from llama_index.core.base.llms.types import TextBlock, ChatMessage, ImageBlock
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.citation_query_engine import (
    CITATION_CHAT_CONTENT_QA_TEMPLATE,
    CITATION_CHAT_CONTENT_REFINE_TEMPLATE,
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
    CitationQueryEngine,
)
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import (
    MediaResource,
    MetadataMode,
    Node,
    NodeWithScore,
    TextNode,
)


_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_init_multimodal__wires_citation_chat_templates():
    engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(is_chat_model=True),
        multimodal=True,
    )
    synth = engine._response_synthesizer
    assert synth._multimodal is True
    assert synth._chat_content_qa_template is CITATION_CHAT_CONTENT_QA_TEMPLATE
    assert synth._chat_content_refine_template is CITATION_CHAT_CONTENT_REFINE_TEMPLATE


def test_init_default__not_multimodal_uses_text_citation_templates():
    engine = CitationQueryEngine(retriever=MagicMock(), llm=MockLLM())
    synth = engine._response_synthesizer
    assert synth._multimodal is False
    assert synth._text_qa_template is CITATION_QA_TEMPLATE
    assert synth._refine_template is CITATION_REFINE_TEMPLATE


def test_create_citation_nodes_text__prefixes_each_chunk_with_source_label():
    query_engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(),
        multimodal=False,
        citation_chunk_size=32,
        citation_chunk_overlap=0,
        text_splitter=SentenceSplitter(chunk_size=32, chunk_overlap=0),
    )
    src = NodeWithScore(node=TextNode(text="A. " * 200), score=0.5)
    nodes = query_engine._create_citation_nodes([src])
    assert len(nodes) >= 2  # got chunked
    for i, nws in enumerate(nodes, start=1):
        assert nws.node.get_content().startswith(f"Source {i}:\n")


def test_create_multimodal_citation_nodes__text_only_chunks_with_source_labels():
    query_engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(is_chat_model=True),
        multimodal=True,
        citation_chunk_size=32,
        citation_chunk_overlap=0,
        text_splitter=SentenceSplitter(chunk_size=32, chunk_overlap=0),
    )
    src = NodeWithScore(
        node=Node(
            text_resource=MediaResource(
                text="The quick brown fox jumps over the lazy dog. " * 30
            )
        ),
        score=0.9,
    )
    nodes = query_engine._create_multimodal_citation_nodes([src])
    assert len(nodes) > 1
    for i, nws in enumerate(nodes, start=1):
        assert nws.node.text_resource is not None
        assert nws.node.text_resource.text.startswith(f"Source {i}:\n")
        # only a text block per chunk since the source has no media
        blocks = nws.node.get_content_blocks(metadata_mode=MetadataMode.LLM)
        assert [b.block_type for b in blocks] == ["text"]


def test_create_multimodal_citation_nodes__image_gets_its_own_source():
    query_engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(is_chat_model=True),
        multimodal=True,
        citation_chunk_size=32,
        citation_chunk_overlap=0,
        text_splitter=SentenceSplitter(chunk_size=32, chunk_overlap=0),
    )
    src = NodeWithScore(
        node=Node(
            text_resource=MediaResource(text="Lorem ipsum dolor sit amet. " * 5),
            image_resource=MediaResource(data=_PNG_1x1, mimetype="image/png"),
        ),
        score=0.9,
    )
    nodes = query_engine._create_multimodal_citation_nodes([src])
    assert len(nodes) == 2
    first_nws, last_nws = nodes
    first_blocks = first_nws.node.get_content_blocks(metadata_mode=MetadataMode.LLM)
    last_blocks = last_nws.node.get_content_blocks(metadata_mode=MetadataMode.LLM)
    first_block_types = [b.block_type for b in first_blocks]
    last_block_types = [b.block_type for b in last_blocks]
    assert first_block_types == ["text"]
    assert last_block_types == ["text", "image"]
    # Every chunk has a source label as the first block
    for i, nws in enumerate(nodes, start=1):
        blocks = nws.node.get_content_blocks(metadata_mode=MetadataMode.LLM)
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text.startswith(f"Source {i}:")


def test_create_multimodal_citation_nodes__preserves_source_node_metadata():
    query_engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(is_chat_model=True),
        multimodal=True,
        citation_chunk_size=64,
        citation_chunk_overlap=0,
        text_splitter=SentenceSplitter(chunk_size=64, chunk_overlap=0),
    )
    src = NodeWithScore(
        node=Node(
            id_="src-1",
            text_resource=MediaResource(text="abc " * 50),
            metadata={"file": "doc1.txt"},
        ),
        score=0.42,
    )
    nodes = query_engine._create_multimodal_citation_nodes([src])
    for nws in nodes:
        assert nws.node.metadata == {"file": "doc1.txt"}
        assert nws.score == 0.42


def test_query__synthesizer_called_multimodal_citation_nodes():
    query_engine = CitationQueryEngine(
        retriever=MagicMock(),
        llm=MockLLM(is_chat_model=True),
        multimodal=True,
        citation_chunk_size=64,
        citation_chunk_overlap=0,
        text_splitter=SentenceSplitter(chunk_size=64, chunk_overlap=0),
    )
    src = NodeWithScore(
        node=Node(
            id_="src-1",
            text_resource=MediaResource(text="abc " * 50),
            image_resource=MediaResource(data=_PNG_1x1, mimetype="image/png"),
        )
    )
    query_engine.retriever.retrieve.return_value = [src]
    with patch.object(
        CompactAndRefine,
        "get_response_from_messages",
        return_value="mock response",
    ) as mock_get_response_from_messages:
        response = query_engine.query("mock query")

    assert str(response) == "mock response"
    assert mock_get_response_from_messages.call_count == 1
    assert mock_get_response_from_messages.call_args_list[0].kwargs[
        "message_chunks"
    ] == [
        ChatMessage(blocks=[TextBlock(text="Source 1:\n" + "abc " * 50 + "\n")]),
        ChatMessage(blocks=[TextBlock(text="Source 2:\n"), ImageBlock(image=_PNG_1x1)]),
    ]
