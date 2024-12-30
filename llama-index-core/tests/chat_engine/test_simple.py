from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.simple import SimpleChatEngine


def test_simple_chat_engine() -> None:
    engine = SimpleChatEngine.from_defaults()

    engine.reset()
    response = engine.chat("Test message 1")
    assert str(response) == "user: Test message 1\nassistant: "

    response = engine.chat("Test message 2")
    assert (
        str(response)
        == "user: Test message 1\nassistant: user: Test message 1\nassistant: \n"
        "user: Test message 2\nassistant: "
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "user: Test message 3\nassistant: "


def test_simple_chat_engine_with_init_history() -> None:
    engine = SimpleChatEngine.from_defaults(
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
