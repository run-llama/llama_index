import secrets
import string

import pytest
from llama_index.llms import ChatMessage, MessageRole
from llama_index.memory.redis_chat_memory_buffer import RedisChatMemoryBuffer
from llama_index.utils import get_tokenizer

tokenizer = get_tokenizer()

USER_CHAT_MESSAGE = ChatMessage(role=MessageRole.USER, content="first message")
USER_CHAT_MESSAGE_TOKENS = len(tokenizer(str(USER_CHAT_MESSAGE.content)))
SECOND_USER_CHAT_MESSAGE = ChatMessage(role=MessageRole.USER, content="second message")
SECOND_USER_CHAT_MESSAGE_TOKENS = len(tokenizer(str(SECOND_USER_CHAT_MESSAGE.content)))
ASSISTANT_CHAT_MESSAGE = ChatMessage(role=MessageRole.ASSISTANT, content="first answer")
ASSISTANT_CHAT_MESSAGE_TOKENS = len(tokenizer(str(ASSISTANT_CHAT_MESSAGE.content)))
SECOND_ASSISTANT_CHAT_MESSAGE = ChatMessage(
    role=MessageRole.USER, content="second answer"
)
SECOND_ASSISTANT_CHAT_MESSAGE_TOKENS = len(
    tokenizer(str(SECOND_ASSISTANT_CHAT_MESSAGE.content))
)


def generate_random_id():
    alphabet = string.ascii_letters + string.digits
    "".join(secrets.choice(alphabet) for _ in range(5))


# use a database of /1 to avoid conflicts with other databases
redis_url = "redis://localhost:6379/1"


def test_put_get() -> None:
    # Given one message in the memory without limit
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        chat_history=[USER_CHAT_MESSAGE],
    )

    # When I get the chat history from the memory
    history = memory.get()

    # Then the history should contain the message
    assert len(history) == 1
    assert history[0].content == USER_CHAT_MESSAGE.content


def test_get_when_initial_tokens_less_than_limit_returns_history() -> None:
    # Given some initial tokens much smaller than token_limit and message tokens
    initial_tokens = 5

    # Given a user message
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=1000,
        chat_history=[USER_CHAT_MESSAGE],
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should contain the message
    assert len(history) == 1
    assert history[0] == USER_CHAT_MESSAGE


def test_get_when_initial_tokens_exceed_limit_raises_value_error() -> None:
    # Given some initial tokens exceeding token_limit
    initial_tokens = 50
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=initial_tokens - 1,
    )

    # When I get the chat history from the memory
    with pytest.raises(ValueError) as error:
        memory.get(initial_tokens)

    # Then a value error should be raised
    assert str(error.value) == "Initial token count exceeds token limit"


def test_get_when_initial_tokens_same_as_limit_removes_message() -> None:
    # Given some initial tokens equal to the token_limit
    initial_tokens = 5

    # Given a user message
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=initial_tokens,
        chat_history=[USER_CHAT_MESSAGE],
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should be empty
    assert len(history) == 0


def test_get_when_space_for_assistant_message_removes_assistant_message_at_start_of_history() -> (
    None
):
    # Given some initial tokens equal to the token_limit minus the user message
    token_limit = 5
    initial_tokens = token_limit - USER_CHAT_MESSAGE_TOKENS

    # Given a user message and an assistant answer
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=token_limit,
        chat_history=[USER_CHAT_MESSAGE, ASSISTANT_CHAT_MESSAGE],
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should be empty
    assert len(history) == 0


def test_get_when_space_for_second_message_and_answer_removes_only_first_message_and_answer() -> (
    None
):
    # Given some initial tokens equal to the token_limit minus one message and one answer
    token_limit = 5
    initial_tokens = (
        token_limit - USER_CHAT_MESSAGE_TOKENS - ASSISTANT_CHAT_MESSAGE_TOKENS
    )

    # Given two user messages and two assistant answers
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=token_limit,
        chat_history=[
            USER_CHAT_MESSAGE,
            ASSISTANT_CHAT_MESSAGE,
            SECOND_USER_CHAT_MESSAGE,
            SECOND_ASSISTANT_CHAT_MESSAGE,
        ],
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should contain the second message and the second answer
    assert len(history) == 2
    assert history[0] == SECOND_USER_CHAT_MESSAGE
    assert history[1] == SECOND_ASSISTANT_CHAT_MESSAGE


def test_get_when_space_for_all_but_first_message_removes_first_message_and_answer() -> (
    None
):
    # Given some initial tokens equal to the token_limit minus one message and one answer
    token_limit = 10
    history_tokens = (
        ASSISTANT_CHAT_MESSAGE_TOKENS
        + USER_CHAT_MESSAGE_TOKENS
        + SECOND_ASSISTANT_CHAT_MESSAGE_TOKENS
    )
    initial_tokens = token_limit - history_tokens

    # Given two user messages and two assistant answers
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        token_limit=token_limit,
        chat_history=[
            USER_CHAT_MESSAGE,
            ASSISTANT_CHAT_MESSAGE,
            SECOND_USER_CHAT_MESSAGE,
            SECOND_ASSISTANT_CHAT_MESSAGE,
        ],
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should contain the second message and the second answer
    assert len(history) == 2
    assert history[0] == SECOND_USER_CHAT_MESSAGE
    assert history[1] == SECOND_ASSISTANT_CHAT_MESSAGE


def test_set() -> None:
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        chat_history=[USER_CHAT_MESSAGE],
    )

    memory.put(USER_CHAT_MESSAGE)

    assert len(memory.get()) == 2

    memory.set([USER_CHAT_MESSAGE])
    assert len(memory.get()) == 1


def test_max_tokens() -> None:
    memory = RedisChatMemoryBuffer.from_defaults(
        session_id=generate_random_id(),
        redis_url=redis_url,
        chat_history=[USER_CHAT_MESSAGE],
        token_limit=5,
    )

    memory.put(USER_CHAT_MESSAGE)
    assert len(memory.get()) == 2

    # do we limit properly
    memory.put(USER_CHAT_MESSAGE)
    memory.put(USER_CHAT_MESSAGE)
    assert len(memory.get()) == 2

    # does get_all work
    assert len(memory.get_all()) == 4

    # does get return in the correct order?
    memory.put(ChatMessage(role=MessageRole.USER, content="test message2"))
    assert memory.get()[-1].content == "test message2"
    assert len(memory.get()) == 2
