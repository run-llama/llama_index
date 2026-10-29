import pytest

from llama_index.core.extractors import DocumentContextExtractor
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from llama_index.core.schema import Document, NodeRelationship, TextNode
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore


@pytest.fixture()
def mock_llm():
    class CustomMockLLM(MockLLM):
        def chat(self, messages, **kwargs):
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    blocks=[
                        {
                            "text": f"Context for the provided chunk",
                            "block_type": "text",
                        }
                    ],
                )
            )

    return CustomMockLLM()


@pytest.fixture()
def sample_documents():
    return [
        Document(
            text="This is chapter 1. It contains important information. This is a test document.",
            metadata={"title": "Doc 1"},
        ),
        Document(
            text="Chapter 2 builds on previous concepts. It introduces new ideas. More test content here.",
            metadata={"title": "Doc 2"},
        ),
    ]


@pytest.fixture()
def create_text_nodes():
    def _create_nodes(document, texts):
        doc_info = document.as_related_node_info()
        return [
            TextNode(
                text=text,
                metadata={},
                relationships={NodeRelationship.SOURCE: doc_info},
            )
            for text in texts
        ]

    return _create_nodes


@pytest.fixture()
def docstore(sample_documents):
    docstore = SimpleDocumentStore()
    for doc in sample_documents:
        docstore.add_documents([doc])
    return docstore


@pytest.fixture()
def context_extractor(docstore, mock_llm):
    return DocumentContextExtractor(
        docstore=docstore,
        llm=mock_llm,
        max_context_length=1000,
        max_output_tokens=100,
        oversized_document_strategy="error",
    )


@pytest.mark.asyncio
async def test_context_extraction_basic(
    context_extractor, sample_documents, create_text_nodes
):
    doc = sample_documents[0]
    nodes = create_text_nodes(
        doc, ["This is chapter 1.", "It contains important information."]
    )

    try:
        metadata_list = await context_extractor.aextract(nodes)
        print("METADATA LIST: ", metadata_list)

        if metadata_list is None:
            raise ValueError("context_extractor.aextract() returned None")

        assert len(metadata_list) == len(nodes)
        for metadata in metadata_list:
            assert "context" in metadata
            assert metadata["context"] == "Context for the provided chunk"

    except Exception as e:
        print(f"Error during extraction: {e!s}")
        raise


def test_invalid_oversized_strategy():
    with pytest.raises(ValueError):
        DocumentContextExtractor(
            docstore=SimpleDocumentStore(),
            llm=MockLLM(),
            max_context_length=1000,
            max_output_tokens=100,
            oversized_document_strategy="invalid_strategy",
        )


@pytest.mark.asyncio
async def test_context_extraction_oversized_document(create_text_nodes):
    large_doc = Document(
        text="This is a very long document. " * 1000, metadata={"title": "Large Doc"}
    )

    docstore = SimpleDocumentStore()
    docstore.add_documents([large_doc])

    extractor = DocumentContextExtractor(
        docstore=docstore,
        llm=MockLLM(),
        max_context_length=100,  # Small limit to trigger error
        max_output_tokens=50,
        oversized_document_strategy="error",
    )

    nodes = create_text_nodes(large_doc, ["This is a test chunk."])

    with pytest.raises(ValueError):
        await extractor.aextract(nodes)


@pytest.mark.asyncio
async def test_context_extraction_custom_prompt(
    docstore, mock_llm, sample_documents, create_text_nodes
):
    custom_prompt = "Generate a detailed context for this chunk:"
    extractor = DocumentContextExtractor(
        docstore=docstore,
        llm=mock_llm,
        prompt=DocumentContextExtractor.ORIGINAL_CONTEXT_PROMPT,
        max_context_length=1000,
        max_output_tokens=100,
    )

    nodes = create_text_nodes(sample_documents[0], ["Test chunk"])

    metadata_list = await extractor.aextract(nodes)
    assert len(metadata_list) == 1
    assert "context" in metadata_list[0]


@pytest.mark.asyncio
async def test_multiple_documents_context(
    context_extractor, sample_documents, create_text_nodes
):
    # Create nodes from different documents
    nodes = create_text_nodes(
        sample_documents[0], ["This is chapter 1."]
    ) + create_text_nodes(
        sample_documents[1], ["Chapter 2 builds on previous concepts."]
    )

    metadata_list = await context_extractor.aextract(nodes)
    assert len(metadata_list) == 2
    for metadata in metadata_list:
        assert "context" in metadata


def test_default_llm_from_settings(docstore, mock_llm, monkeypatch):
    from llama_index.core.settings import Settings

    monkeypatch.setattr(Settings, "_llm", mock_llm)
    extractor = DocumentContextExtractor(docstore=docstore)
    assert extractor.llm is mock_llm


@pytest.mark.asyncio
async def test_get_document_reraises_unexpected_value_error(docstore, mock_llm, monkeypatch):
    extractor = DocumentContextExtractor(docstore=docstore, llm=mock_llm)

    async def _raise_corrupt(_doc_id: str):
        raise ValueError("corrupted document payload")

    monkeypatch.setattr(docstore, "aget_document", _raise_corrupt)
    with pytest.raises(ValueError, match="corrupted document payload"):
        await extractor._get_document("doc-1")


@pytest.mark.asyncio
async def test_aextract_caches_parent_document_and_handles_duplicate_node_ids(
    docstore, mock_llm, sample_documents, create_text_nodes, monkeypatch
):
    extractor = DocumentContextExtractor(docstore=docstore, llm=mock_llm)
    doc = sample_documents[0]
    nodes = create_text_nodes(doc, ["Chunk A", "Chunk B", "Chunk C"])
    # Give two nodes the same node_id to verify positional metadata mapping
    nodes[1].node_id = nodes[0].node_id

    fetch_count = 0
    orig_aget_document = docstore.aget_document

    async def _counted_aget_document(doc_id: str, *args, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return await orig_aget_document(doc_id, *args, **kwargs)

    monkeypatch.setattr(docstore, "aget_document", _counted_aget_document)

    metadata_list = await extractor.aextract(nodes)
    assert fetch_count == 1
    assert len(metadata_list) == 3
    for metadata in metadata_list:
        assert metadata == {"context": "Context for the provided chunk"}

