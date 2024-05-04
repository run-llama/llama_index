from unittest.mock import MagicMock, PropertyMock
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)

from llama_index.agent.introspective import IntrospectiveAgentWorker

PRINT_CHAT_HISTORY = True


def test_introspective_agent() -> None:
    # Arrange
    # mock main agent
    mock_main_agent_worker = MagicMock()
    mock_main_agent = MagicMock()
    mock_main_memory = PropertyMock(
        return_value=ChatMemoryBuffer.from_defaults(
            chat_history=[
                ChatMessage(
                    content="You are a helpful assistant.", role=MessageRole.SYSTEM
                ),
                ChatMessage(content="A mock user task.", role=MessageRole.USER),
                ChatMessage(
                    content="This is a mock initial response.",
                    role=MessageRole.ASSISTANT,
                ),
            ]
        )
    )
    type(mock_main_agent).memory = mock_main_memory
    mock_main_agent.chat.return_value = AgentChatResponse(
        response="This is a mock initial response."
    )
    mock_main_agent_worker.as_agent.return_value = mock_main_agent

    # mock reflective agent
    mock_reflective_agent_worker = MagicMock()
    mock_reflective_agent = MagicMock()
    mock_reflective_memory = PropertyMock(
        return_value=ChatMemoryBuffer.from_defaults(
            chat_history=[
                ChatMessage(
                    content="You are a helpful assistant.", role=MessageRole.SYSTEM
                ),
                ChatMessage(content="A mock user task.", role=MessageRole.USER),
                ChatMessage(
                    content="This is a mock initial response.",
                    role=MessageRole.ASSISTANT,
                ),
                ChatMessage(
                    content="This is a mock reflection.", role=MessageRole.USER
                ),
                ChatMessage(
                    content="This is a mock corrected response!",
                    role=MessageRole.ASSISTANT,
                ),
            ]
        )
    )
    type(mock_reflective_agent).memory = mock_reflective_memory
    mock_reflective_agent.chat.return_value = AgentChatResponse(
        response="This is a mock corrected response!"
    )
    mock_reflective_agent_worker.as_agent.return_value = mock_reflective_agent

    # build introspective agent
    worker = IntrospectiveAgentWorker.from_defaults(
        reflective_agent_worker=mock_reflective_agent_worker,
        main_agent_worker=mock_main_agent_worker,
    )
    chat_history = [
        ChatMessage(
            content="You are a helpful assistant.",
            role=MessageRole.SYSTEM,
        )
    ]
    agent = worker.as_agent(chat_history=chat_history)

    # Act
    response = agent.chat("A mock user task.")  # reflect on current response

    # Assert
    if PRINT_CHAT_HISTORY:
        for msg in agent.chat_history:
            print(str(msg))
            print()
    assert response.response == "This is a mock corrected response!"
    assert len(agent.chat_history) == 3  # (system, user, asst)
