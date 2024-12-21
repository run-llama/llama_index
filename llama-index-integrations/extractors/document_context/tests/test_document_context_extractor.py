
# this goes here: C:\Users\cklap\llama_index\llama-index-core\tests\node_parser\metadata_extractor.py
import pytest
from llama_index.core.schema import Document, Node
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.llms import MockLLM
from llama_index.core.llms import ChatMessage
from llama_index.integrations.extractors.document_context import DocumentContextExtractor

@pytest.fixture
def mock_llm():
    class CustomMockLLM(MockLLM):
        def chat(self, messages, **kwargs):
            # Mock response that simulates context generation
            return ChatMessage(
                role="assistant", 
                blocks=[{"text": f"Context for the provided chunk", "block_type": "text"}]
            )
    
    return CustomMockLLM()

@pytest.fixture
def sample_documents():
    # Create some test documents
    docs = [
        Document(
            text="This is chapter 1. It contains important information. This is a test document.",
            metadata={"title": "Doc 1"}
        ),
        Document(
            text="Chapter 2 builds on previous concepts. It introduces new ideas. More test content here.",
            metadata={"title": "Doc 2"}
        )
    ]
    return docs

@pytest.fixture
def docstore(sample_documents):
    # Initialize docstore with sample documents
    docstore = SimpleDocumentStore()
    for doc in sample_documents:
        docstore.add_documents([doc])
    return docstore

@pytest.fixture
def context_extractor(docstore, mock_llm):
    return DocumentContextExtractor(
        docstore=docstore,
        llm=mock_llm,
        max_context_length=1000,
        max_contextual_tokens=100,
        oversized_document_strategy="truncate_first"
    )

@pytest.mark.asyncio
async def test_context_extraction_basic(context_extractor, sample_documents):
    # Create nodes from the first document
    nodes = [
        Node(
            text="This is chapter 1.",
            metadata={},
            source_node=sample_documents[0]
        ),
        Node(
            text="It contains important information.",
            metadata={},
            source_node=sample_documents[0]
        )
    ]
    
    # Extract context
    metadata_list = await context_extractor.aextract(nodes)
    
    # Verify each node got context
    assert len(metadata_list) == len(nodes)
    for metadata in metadata_list:
        assert "context" in metadata
        assert metadata["context"] == "Context for the provided chunk"

@pytest.mark.asyncio
async def test_context_extraction_oversized_document():
    # Create a very large document
    large_doc = Document(
        text="This is a very long document. " * 1000,
        metadata={"title": "Large Doc"}
    )
    
    docstore = SimpleDocumentStore()
    docstore.add_documents([large_doc])
    
    extractor = DocumentContextExtractor(
        docstore=docstore,
        llm=MockLLM(),
        max_context_length=100,  # Small limit to trigger truncation
        max_contextual_tokens=50,
        oversized_document_strategy="truncate_first"
    )
    
    node = Node(
        text="This is a test chunk.",
        metadata={},
        source_node=large_doc
    )
    
    # Should not raise an error due to truncation strategy
    metadata_list = await extractor.aextract([node])
    assert len(metadata_list) == 1

@pytest.mark.asyncio
async def test_context_extraction_custom_prompt(docstore, mock_llm):
    custom_prompt = "Generate a detailed context for this chunk:"
    extractor = DocumentContextExtractor(
        docstore=docstore,
        llm=mock_llm,
        prompts=[custom_prompt],
        max_context_length=1000,
        max_contextual_tokens=100
    )
    
    node = Node(
        text="Test chunk",
        metadata={},
        source_node=next(iter(docstore.docs.values()))
    )
    
    metadata_list = await extractor.aextract([node])
    assert len(metadata_list) == 1
    assert "context" in metadata_list[0]

@pytest.mark.asyncio
async def test_multiple_documents_context(context_extractor, sample_documents):
    # Create nodes from different documents
    nodes = [
        Node(
            text="This is chapter 1.",
            metadata={},
            source_node=sample_documents[0]
        ),
        Node(
            text="Chapter 2 builds on previous concepts.",
            metadata={},
            source_node=sample_documents[1]
        )
    ]
    
    metadata_list = await context_extractor.aextract(nodes)
    assert len(metadata_list) == 2
    for metadata in metadata_list:
        assert "context" in metadata

def test_invalid_oversized_strategy():
    with pytest.raises(ValueError):
        DocumentContextExtractor(
            docstore=SimpleDocumentStore(),
            llm=MockLLM(),
            max_context_length=1000,
            max_contextual_tokens=100,
            oversized_document_strategy="invalid_strategy"
        )