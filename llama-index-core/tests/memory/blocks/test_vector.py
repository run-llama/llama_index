import pytest
from typing import Any, Dict, List, Sequence

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.memory.memory_blocks.vector import VectorMemoryBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class MockVectorStore(BasePydanticVectorStore):
    """Mock vector store for testing."""

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(self):
        super().__init__()
        self._nodes = {}

    @property
    def client(self) -> Any:
        return self

    @property
    def nodes(self) -> Dict[str, BaseNode]:
        return self._nodes

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Add nodes to vector store."""
        ids = []
        for node in nodes:
            self._nodes[node.id_] = node
            ids.append(node.id_)
        return ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Async add nodes to vector store."""
        return self.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes with ref_doc_id."""
        for node_id in list(self._nodes.keys()):
            if self._nodes[node_id].ref_doc_id == ref_doc_id:
                del self._nodes[node_id]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        # For simplicity, return all nodes
        nodes = list(self._nodes.values())
        if query.similarity_top_k and len(nodes) > query.similarity_top_k:
            nodes = nodes[: query.similarity_top_k]

        # Simulate similarity scores
        similarities = [0.9 - 0.1 * i for i in range(len(nodes))]
        ids = [node.id_ for node in nodes]

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async query vector store."""
        return self.query(query, **kwargs)


class MockNodePostprocessor(BaseNodePostprocessor):
    """Mock node postprocessor for testing."""

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query: Any = None
    ) -> List[NodeWithScore]:
        """Add a prefix to each node's text."""
        for node in nodes:
            if isinstance(node.node, TextNode):
                node.node.text = f"PROCESSED: {node.node.text}"
        return nodes


@pytest.fixture
def mock_embedding():
    """Create a mock embedding model."""
    return MockEmbedding(embed_dim=10)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return MockVectorStore()


@pytest.fixture
def vector_memory_block(
    mock_vector_store: MockVectorStore, mock_embedding: MockEmbedding
):
    """Create a vector memory block."""
    return VectorMemoryBlock(
        vector_store=mock_vector_store,
        embed_model=mock_embedding,
        similarity_top_k=2,
    )


@pytest.mark.asyncio
async def test_vector_memory_block_put(vector_memory_block: VectorMemoryBlock):
    """Test putting messages in the vector memory block."""
    # Create messages
    messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing well, thank you for asking!"),
    ]

    # Put messages in memory
    await vector_memory_block.aput(messages=messages)

    # Check that messages were added to vector store
    assert len(vector_memory_block.vector_store.nodes) == 1

    # Check node content contains both messages
    node = next(iter(vector_memory_block.vector_store.nodes.values()))
    assert "<message role='user'>Hello, how are you?</message>" in node.text
    assert (
        "<message role='assistant'>I'm doing well, thank you for asking!</message>"
        in node.text
    )


@pytest.mark.asyncio
async def test_vector_memory_block_get(vector_memory_block: VectorMemoryBlock):
    """Test getting messages from the vector memory block."""
    # Create and store some messages
    history_messages = [
        ChatMessage(role="user", content="What's the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris."),
        ChatMessage(role="user", content="What about Germany?"),
        ChatMessage(role="assistant", content="The capital of Germany is Berlin."),
    ]

    await vector_memory_block.aput(messages=history_messages)

    # Create a new query
    query_messages = [ChatMessage(role="user", content="Tell me about Paris.")]

    # Get relevant information
    result = await vector_memory_block.aget(messages=query_messages)

    # Check that we got a result
    assert result != ""
    assert "capital of France is Paris" in result


@pytest.mark.asyncio
async def test_empty_messages(vector_memory_block: VectorMemoryBlock):
    """Test with empty messages."""
    # Test empty get
    result = await vector_memory_block.aget(messages=[])
    assert result == ""

    # Test empty put
    await vector_memory_block.aput(messages=[])
    assert len(vector_memory_block.vector_store.nodes) == 0


@pytest.mark.asyncio
async def test_message_without_text(vector_memory_block: VectorMemoryBlock):
    """Test with a message that has no text blocks."""
    # Create a message with no text blocks
    message = ChatMessage(role="user", content=None, blocks=[])

    # Put the message in memory
    await vector_memory_block.aput(messages=[message])

    # Check that nothing was added
    assert len(vector_memory_block.vector_store.nodes) == 0


@pytest.mark.asyncio
async def test_retrieval_context_window(
    mock_vector_store: MockVectorStore, mock_embedding: MockEmbedding
):
    """Test the retrieval_context_window parameter."""
    # Create a memory block with a specific context window
    memory_block = VectorMemoryBlock(
        vector_store=mock_vector_store,
        embed_model=mock_embedding,
        retrieval_context_window=2,
        similarity_top_k=2,
    )

    # Create and store some messages
    history_messages = [
        ChatMessage(role="user", content="What's your name?"),
        ChatMessage(role="assistant", content="I'm an AI assistant."),
        ChatMessage(role="user", content="What's the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris."),
    ]

    await memory_block.aput(messages=history_messages)

    # Create a query with multiple messages
    query_messages = [
        ChatMessage(role="user", content="What about the UK?"),
        ChatMessage(role="assistant", content="The capital of the UK is London."),
        ChatMessage(role="user", content="And Germany?"),
    ]

    # The retrieval should only use the last 2 messages
    result = await memory_block.aget(messages=query_messages)

    # Check that we got a result
    assert result != ""
    # The result should be more related to UK/London than Paris
    # In our mock implementation, it will just return all stored nodes


@pytest.mark.asyncio
async def test_node_postprocessors(
    mock_vector_store: MockVectorStore, mock_embedding: MockEmbedding
):
    """Test node postprocessors."""
    # Create a postprocessor
    postprocessor = MockNodePostprocessor()

    # Create a memory block with the postprocessor
    memory_block = VectorMemoryBlock(
        vector_store=mock_vector_store,
        embed_model=mock_embedding,
        similarity_top_k=2,
        node_postprocessors=[postprocessor],
    )

    # Create and store some messages
    history_messages = [
        ChatMessage(role="user", content="What's the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris."),
    ]

    await memory_block.aput(messages=history_messages)

    # Create a query
    query_messages = [ChatMessage(role="user", content="Tell me about Paris.")]

    # Get relevant information - this should be processed
    result = await memory_block.aget(messages=query_messages)

    # Check that the result contains the processed prefix
    assert "PROCESSED:" in result


@pytest.mark.asyncio
async def test_format_template(
    mock_vector_store: MockVectorStore, mock_embedding: MockEmbedding
):
    """Test custom format template."""
    # Create a memory block with a custom format template
    custom_template = RichPromptTemplate("Relevant context: {{ text }}")

    memory_block = VectorMemoryBlock(
        vector_store=mock_vector_store,
        embed_model=mock_embedding,
        similarity_top_k=2,
        format_template=custom_template,
    )

    # Create and store some messages
    history_messages = [
        ChatMessage(role="user", content="What's the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris."),
    ]

    await memory_block.aput(messages=history_messages)

    # Create a query
    query_messages = [ChatMessage(role="user", content="Tell me about Paris.")]

    # Get relevant information with custom format
    result = await memory_block.aget(messages=query_messages)

    # Check that the result contains our custom prefix
    assert result.startswith("Relevant context:")
    assert "capital of France is Paris" in result
