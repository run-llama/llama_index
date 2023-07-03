from llama_index.chat_engine.simple import SimpleChatEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.llms.base import ChatMessage, MessageRole


def test_simple_chat_engine(
    mock_service_context: ServiceContext,
) -> None:
    engine = SimpleChatEngine.from_defaults(service_context=mock_service_context)

    engine.reset()
    response = engine.chat("Test message 1")
    print("1 response: ", response)
    assert str(response) == "user: Test message 1\nassistant: "

    response = engine.chat("Test message 2")
    print("2 response: ", response)
    assert (
        str(response)
        == "user: Test message 1\nassistant: user: Test message 1\nassistant: \n"
        "user: Test message 2\nassistant: "
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "user: Test message 3\nassistant: "


def test_simple_chat_engine_with_init_history(
    mock_service_context: ServiceContext,
) -> None:
    engine = SimpleChatEngine.from_defaults(
        service_context=mock_service_context,
        chat_history=[
            ChatMessage(role=MessageRole.USER, content="test human message"),
            ChatMessage(role=MessageRole.ASSISTANT, content="test ai message"),
        ],
    )

    response = engine.chat("new human message")
    assert (
        str(response) == "user: test human message\nassistant: test ai message\n"
        "user: new human message\nassistant: "
    )
