from unittest.mock import Mock

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.chat_engine.condense_question import (
    CondenseQuestionChatEngine,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer


def test_condense_question_chat_engine(patch_llm_predictor) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

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


def test_condense_question_chat_engine_with_init_history(patch_llm_predictor) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=[
            ChatMessage(role=MessageRole.USER, content="test human message"),
            ChatMessage(role=MessageRole.ASSISTANT, content="test ai message"),
        ],
    )

    response = engine.chat("new human message")
    assert str(response) == (
        "{'question': 'new human message', 'chat_history': 'user: test human "
        "message\\nassistant: test ai message'}"
    )


def test_stream_chat_history_write_completes_on_early_exit() -> None:
    def token_gen():
        for token in ["Hello", " ", "World", "!"]:
            yield token

    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.return_value = StreamingResponse(response_gen=token_gen())
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, llm=MockLLM()
    )
    engine._memory = ChatMemoryBuffer.from_defaults()

    response = engine.stream_chat("Hello!")
    gen = response.response_gen
    for i, _ in enumerate(gen):
        if i >= 2:
            break
    gen.close()
    assert len(engine.chat_history) == 2
    assert engine.chat_history[0].role == MessageRole.USER
    assert engine.chat_history[1].role == MessageRole.ASSISTANT
