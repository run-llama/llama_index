from llama_index.llms import ChatMessage, MessageRole
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer

CHAT_MESSAGE = ChatMessage(role=MessageRole.USER, content="test message")


def test_put_get() -> None:
    memory = ChatMemoryBuffer.from_defaults()

    memory.put(CHAT_MESSAGE)

    assert len(memory.get()) == 1
    assert memory.get()[0].content == CHAT_MESSAGE.content


def test_set() -> None:
    memory = ChatMemoryBuffer.from_defaults(chat_history=[CHAT_MESSAGE])

    memory.put(CHAT_MESSAGE)

    assert len(memory.get()) == 2

    memory.set([CHAT_MESSAGE])
    assert len(memory.get()) == 1


def test_max_tokens() -> None:
    memory = ChatMemoryBuffer.from_defaults(chat_history=[CHAT_MESSAGE], token_limit=5)

    memory.put(ChatMessage(role=MessageRole.USER, content="test message2"))
    assert len(memory.get()) == 2

    # do we limit properly
    memory.put(CHAT_MESSAGE)
    memory.put(CHAT_MESSAGE)
    assert len(memory.get()) == 2

    # does get_all work
    assert len(memory.get_all()) == 4

    # does get return in the correct order?
    assert memory.get()[-1].content == "test message2"
