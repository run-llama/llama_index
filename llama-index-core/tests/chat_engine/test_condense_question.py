from unittest.mock import Mock

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response
from llama_index.core.chat_engine.condense_question import (
    CondenseQuestionChatEngine,
)
from llama_index.core.service_context import ServiceContext


def test_condense_question_chat_engine(
    mock_service_context: ServiceContext,
) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        service_context=mock_service_context,
    )

    engine.reset()
    response = engine.chat("Test message 1")
    assert str(response) == "Test message 1"

    response = engine.chat("Test message 2")
    assert str(response) == (
        "{"
        "'question': 'Test message 2', "
        "'chat_history': 'user: Test message 1\\nassistant: Test message 1'"
        "}"
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "Test message 3"


def test_condense_question_chat_engine_with_init_history(
    mock_service_context: ServiceContext,
) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        service_context=mock_service_context,
        chat_history=[
            ChatMessage(role=MessageRole.USER, content="test human message"),
            ChatMessage(role=MessageRole.ASSISTANT, content="test ai message"),
        ],
    )

    print(engine.chat_history)

    response = engine.chat("new human message")
    assert str(response) == (
        "{'question': 'new human message', 'chat_history': 'user: test human "
        "message\\nassistant: test ai message'}"
    )
