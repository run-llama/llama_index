from typing import Any, List
from unittest.mock import Mock, patch

from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.flow import FlowChatEngine
from llama_index.core.service_context import ServiceContext


def test_flow_chat(
    mock_service_context: ServiceContext,
) -> None:
    context = "Test Node"

    mock_retriever = Mock(spec=BaseRetriever)

    def override_retrieve(query: str) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    text=context,
                    id_="id_000001",
                ),
                score=0.9,
            )
        ]

    mock_retriever.retrieve.side_effect = override_retrieve

    memory = ChatMemoryBuffer.from_defaults(
        chat_history=[], llm=mock_service_context.llm
    )
    prefix_messages = [
        ChatMessage(role=MessageRole.SYSTEM, text="You are a friendly bot.")
    ]
    engine = FlowChatEngine.from_defaults(
        retriever=mock_retriever,
        llm=MockLLM(),
        memory=memory,
        prefix_messages=prefix_messages,
    )
    engine.reset()

    query = "First Query"
    response = engine.chat(query)

    expected_response = (
        "system: \n"
        "user: Context information is below.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        f"Query: {query}\n"
        "Answer: \n"
        "assistant: "
    )

    assert str(response) == expected_response
    assert memory.get_all()[-2].content == query


def test_flow_stream_chat(
    mock_service_context: ServiceContext,
) -> None:
    context = "Test Node"

    mock_retriever = Mock(spec=BaseRetriever)

    def override_retrieve(query: str) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    text=context,
                    id_="id_000001",
                ),
                score=0.9,
            )
        ]

    mock_retriever.retrieve.side_effect = override_retrieve

    memory = ChatMemoryBuffer.from_defaults(
        chat_history=[], llm=mock_service_context.llm
    )
    prefix_messages = [
        ChatMessage(role=MessageRole.SYSTEM, text="You are a friendly bot.")
    ]
    engine = FlowChatEngine.from_defaults(
        retriever=mock_retriever,
        llm=MockLLM(),
        memory=memory,
        prefix_messages=prefix_messages,
    )
    engine.reset()

    query = "First Query"
    response = engine.stream_chat(query)
    assert isinstance(response, StreamingAgentChatResponse)

    # exhaust stream
    for delta in response.response_gen:
        continue

    expected_response = (
        "system: \n"
        "user: Context information is below.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        f"Query: {query}\n"
        "Answer: \n"
        "assistant:"
    )

    assert response.response == expected_response
    assert memory.get_all()[-2].content == query
