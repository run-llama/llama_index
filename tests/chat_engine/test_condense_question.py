from unittest.mock import Mock
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.response.schema import Response


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
    assert str(response) == "{'question': 'Test message 1', 'chat_history': ''}"

    response = engine.chat("Test message 2")
    assert str(response) == (
        "{'question': 'Test message 2', 'chat_history': \"\\nHuman: Test message 1"
        "\\nAssistant: {'question': 'Test message 1', 'chat_history': ''}\"}"
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "{'question': 'Test message 3', 'chat_history': ''}"


def test_condense_question_chat_engine_with_init_history(
    mock_service_context: ServiceContext,
) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        service_context=mock_service_context,
        chat_history=[("test human message", "test ai message")],
    )

    response = engine.chat("new human message")
    assert str(response) == (
        "{'question': 'new human message', 'chat_history': '\\nHuman: test human "
        "message\\nAssistant: test ai message'}"
    )
