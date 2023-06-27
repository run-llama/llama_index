from llama_index.chat_engine.simple import SimpleChatEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.llms.base import ChatMessage


def test_simple_chat_engine(
    mock_service_context: ServiceContext,
) -> None:
    engine = SimpleChatEngine.from_defaults(service_context=mock_service_context)

    engine.reset()
    response = engine.chat("Test message 1")
    assert str(response) == ":Test message 1"

    response = engine.chat("Test message 2")
    assert (
        str(response)
        == "\nuser: Test message 1\nassistant: :Test message 1:Test message 2"
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == ":Test message 3"


def test_simple_chat_engine_with_init_history(
    mock_service_context: ServiceContext,
) -> None:
    engine = SimpleChatEngine.from_defaults(
        service_context=mock_service_context,
        chat_history=[
            ChatMessage(role="user", content="test human message"),
            ChatMessage(role="assistant", content="test ai message"),
        ],
    )

    response = engine.chat("new human message")
    assert (
        str(response)
        == "\nuser: test human message\nassistant: test ai message\nuser: new human message\nassistant: "
    )
