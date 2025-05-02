from typing import List, Sequence, Any
import pickle

import pytest

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.core.memory.chat_summary_memory_buffer import (
    ChatSummaryMemoryBuffer,
)
from llama_index.core.utils import get_tokenizer

tokenizer = get_tokenizer()


def _get_role_alternating_order(i: int):
    if i % 2 == 0:
        return MessageRole.USER
    return MessageRole.ASSISTANT


try:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    openai_installed = True
except ImportError:
    openai_installed = False

if openai_installed:
    USER_CHAT_MESSAGE = ChatMessage(role=MessageRole.USER, content="first message")
    ASSISTANT_CHAT_MESSAGE = ChatMessage(
        role=MessageRole.ASSISTANT, content="first answer"
    )
    ASSISTANT_TOOL_CALLING_MESSAGE = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=None,
        additional_kwargs={
            "tool_calls": [
                ChatCompletionMessageToolCall(
                    id="call_Opq33YZVi0usHbNcrvEYN9QO",
                    function=Function(arguments='{"a":363,"b":42}', name="add"),
                    type="function",
                )
            ]
        },
    )
    TOOL_CHAT_MESSAGE = ChatMessage(role=MessageRole.TOOL, content="first tool")
    USER_CHAT_MESSAGE_TOKENS = len(tokenizer(str(USER_CHAT_MESSAGE.content)))
    LONG_USER_CHAT_MESSAGE = ChatMessage(
        role=MessageRole.USER,
        content="".join(
            ["This is a message that is longer than the proposed token length"] * 10
        ),
    )
    LONG_RUNNING_CONVERSATION = [
        ChatMessage(role=_get_role_alternating_order(i), content=f"Message {i}")
        for i in range(6)
    ]
    LONG_USER_CHAT_MESSAGE_TOKENS = len(tokenizer(str(LONG_USER_CHAT_MESSAGE.content)))


class MockSummarizerLLM(MockLLM):
    _i: int = PrivateAttr()
    _responses: List[ChatMessage] = PrivateAttr()
    _role_counts: dict = PrivateAttr()

    def __init__(self, responses: List[ChatMessage], max_tokens: int = 512) -> None:
        super().__init__(max_tokens=max_tokens)
        self._i = 0  # call counter, determines which response to return
        self._responses = responses  # list of responses to return
        self._role_counts: dict = {role: 0 for role in MessageRole}

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Count how many messages are going to be summarized for each role
        for role in MessageRole:
            for message in messages:
                self._role_counts[role] += message.content.count(role + ": ")
        del messages

        # For this mockLLM, we assume tokens are separated by spaces
        max_tokens = self.max_tokens
        if self.max_tokens > len(self._responses[self._i].content):
            max_tokens = len(self._responses[self._i].content)
        response_tokens = " ".join(
            self._responses[self._i].content.split(" ")[0:max_tokens]
        )

        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_tokens),
        )
        self._i += 1
        return response

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def get_role_count(self, role: MessageRole):
        return self._role_counts[role]


FIRST_SUMMARY_RESPONSE = "First, the user asked what an LLM was, and the assistant explained the basic ideas."
SECOND_SUMMARY_RESPONSE = (
    "The conversation started about LLMs. It then continued about LlamaIndex."
)


@pytest.fixture()
def summarizer_llm():
    return MockSummarizerLLM(
        responses=[
            ChatMessage(
                content=FIRST_SUMMARY_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
            ChatMessage(
                content=SECOND_SUMMARY_RESPONSE,
                role=MessageRole.ASSISTANT,
            ),
        ]
    )


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_put_get(summarizer_llm) -> None:
    # Given one message with fewer tokens than token_limit
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[USER_CHAT_MESSAGE], llm=summarizer_llm
    )

    # When I get the chat history from the memory
    history = memory.get()

    # Then the history should contain the full message
    assert len(history) == 1
    assert history[0].content == USER_CHAT_MESSAGE.content


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_put_get_summarize_long_message(summarizer_llm) -> None:
    # Given one message with more tokens than token_limit
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[LONG_USER_CHAT_MESSAGE],
        token_limit=2,
        llm=summarizer_llm,
    )

    # When I get the chat history from the memory
    history = memory.get()

    # Then the history should contain the summarized message
    assert len(history) == 1
    assert history[0].content == FIRST_SUMMARY_RESPONSE


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_put_get_summarize_message_with_tool_call(summarizer_llm) -> None:
    # Given one message with more tokens than token_limit and tool calls
    # This case test 2 things:
    #   1. It can summarize the ASSISTANT_TOOL_CALLING_MESSAGE with content=None (Issue #14014).
    #   2. In `_handle_assistant_and_tool_messages`, when chat_history_full_text only
    #      contains tool calls or assistant messages, it could add them all into
    #      `chat_history_to_be_summarized`, without triggering the IndexError.
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[
            LONG_USER_CHAT_MESSAGE,
            ASSISTANT_TOOL_CALLING_MESSAGE,
            TOOL_CHAT_MESSAGE,
            ASSISTANT_CHAT_MESSAGE,
        ],
        token_limit=LONG_USER_CHAT_MESSAGE_TOKENS,
        llm=summarizer_llm,
    )

    # When I get the chat history from the memory
    history = memory.get()

    # Then the history should contain the summarized message
    assert len(history) == 1
    assert history[0].content == FIRST_SUMMARY_RESPONSE


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_put_get_summarize_part_of_conversation(summarizer_llm) -> None:
    # Given a chat history where only 2 responses fit in the token_limit
    tokens_most_recent_messages = sum(
        [
            len(tokenizer(str(LONG_RUNNING_CONVERSATION[-i].content)))
            for i in range(1, 3)
        ]
    )
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=LONG_RUNNING_CONVERSATION.copy(),
        token_limit=tokens_most_recent_messages,
        llm=summarizer_llm,
    )

    # When I get the chat history from the memory
    history = memory.get()

    # Then the history should contain the full message for the latest two and
    # a summary for the older messages
    assert len(history) == 3
    assert history[0].content == FIRST_SUMMARY_RESPONSE
    assert history[0].role == MessageRole.SYSTEM
    assert history[1].content == "Message 4"
    assert history[2].content == "Message 5"

    # When I add new messages to the history
    memory.put(ChatMessage(role=MessageRole.USER, content="Message 6"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Message 7"))

    # Then the history should re-summarize
    history = memory.get()
    assert len(history) == 3
    assert history[0].content == SECOND_SUMMARY_RESPONSE
    assert history[0].role == MessageRole.SYSTEM
    assert history[1].content == "Message 6"
    assert history[2].content == "Message 7"


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_get_when_initial_tokens_less_than_limit_returns_history() -> None:
    # Given some initial tokens much smaller than token_limit and message tokens
    initial_tokens = 5

    # Given a user message
    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=1000, chat_history=[USER_CHAT_MESSAGE]
    )

    # When I get the chat history from the memory
    history = memory.get(initial_tokens)

    # Then the history should contain the message
    assert len(history) == 1
    assert history[0] == USER_CHAT_MESSAGE


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_get_when_initial_tokens_exceed_limit_raises_value_error() -> None:
    # Given some initial tokens exceeding token_limit
    initial_tokens = 50
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[USER_CHAT_MESSAGE],
        token_limit=initial_tokens - 1,
        count_initial_tokens=True,
    )

    # When I get the chat history from the memory
    with pytest.raises(ValueError) as error:
        memory.get(initial_token_count=initial_tokens)

    # Then a value error should be raised
    assert str(error.value) == "Initial token count exceeds token limit"


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_set() -> None:
    memory = ChatSummaryMemoryBuffer.from_defaults(chat_history=[USER_CHAT_MESSAGE])

    memory.put(USER_CHAT_MESSAGE)

    assert len(memory.get()) == 2

    memory.set([USER_CHAT_MESSAGE])
    assert len(memory.get()) == 1


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_max_tokens_without_summarizer() -> None:
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[USER_CHAT_MESSAGE], token_limit=5
    )

    memory.put(USER_CHAT_MESSAGE)
    assert len(memory.get()) == 2

    # do we limit properly
    memory.put(USER_CHAT_MESSAGE)
    memory.put(USER_CHAT_MESSAGE)
    assert len(memory.get()) == 2

    # In ChatSummaryMemoryBuffer, we overwrite the actual chat history
    assert len(memory.get_all()) == 2

    # does get return in the correct order?
    memory.put(ChatMessage(role=MessageRole.USER, content="test message2"))
    assert memory.get()[-1].content == "test message2"
    assert len(memory.get()) == 2


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_max_tokens_with_summarizer(summarizer_llm) -> None:
    max_tokens = 1
    summarizer_llm.set_max_tokens(max_tokens)
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=summarizer_llm,
        chat_history=[USER_CHAT_MESSAGE],
        token_limit=5,
    )

    # do we limit properly
    memory.put(USER_CHAT_MESSAGE)
    memory.put(USER_CHAT_MESSAGE)
    memory_results = memory.get()
    assert len(memory_results) == 3
    # Oldest message is summarized
    assert memory_results[0].content == " ".join(
        FIRST_SUMMARY_RESPONSE.split(" ")[0:max_tokens]
    )
    assert memory_results[0].role == MessageRole.SYSTEM

    # In ChatSummaryMemoryBuffer, we overwrite the actual chat history
    assert len(memory.get_all()) == 3

    # does get return in the correct order?
    memory.put(ChatMessage(role=MessageRole.USER, content="test message2"))
    memory_results = memory.get()
    assert memory_results[-1].content == "test message2"
    assert len(memory_results) == 3
    # Oldest message is summarized based on the latest information
    assert memory_results[0].content == " ".join(
        SECOND_SUMMARY_RESPONSE.split(" ")[0:max_tokens]
    )


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_assistant_never_first_message(summarizer_llm) -> None:
    chat_history = [
        USER_CHAT_MESSAGE,
        ASSISTANT_CHAT_MESSAGE,
        USER_CHAT_MESSAGE,
        ASSISTANT_CHAT_MESSAGE,
    ]
    tokens_last_3_messages = sum(
        [len(tokenizer(str(chat_history[-i].content))) for i in range(1, 4)]
    )

    # When exactly 3 messages fit the buffer, with first being assistant
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=chat_history,
        llm=summarizer_llm,
        token_limit=tokens_last_3_messages,
        count_initial_tokens=False,
    )

    memory_results = memory.get()
    # the assistant message should be summarized instead of full text
    assert len(memory_results) == 3
    assert summarizer_llm.get_role_count(MessageRole.ASSISTANT) == 1
    assert memory_results[1].role == MessageRole.USER
    assert memory_results[2].role == MessageRole.ASSISTANT


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_assistant_tool_pairs(summarizer_llm) -> None:
    chat_history = [
        USER_CHAT_MESSAGE,
        ASSISTANT_CHAT_MESSAGE,
        TOOL_CHAT_MESSAGE,
        USER_CHAT_MESSAGE,
        ASSISTANT_CHAT_MESSAGE,
    ]
    tokens_last_3_messages = sum(
        [len(tokenizer(str(chat_history[-i].content))) for i in range(1, 4)]
    )

    # When exactly 3 messages fit the buffer, with first being assistant
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=chat_history,
        llm=summarizer_llm,
        token_limit=tokens_last_3_messages,
        count_initial_tokens=False,
    )

    memory_results = memory.get()
    # the tool message should be summarized along with the assistant message
    assert len(memory_results) == 3
    assert summarizer_llm.get_role_count(MessageRole.TOOL) == 1
    assert memory_results[1].role == MessageRole.USER
    assert memory_results[2].role == MessageRole.ASSISTANT


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_string_save_load(summarizer_llm) -> None:
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=summarizer_llm,
        chat_history=[USER_CHAT_MESSAGE],
        token_limit=5,
        summarize_prompt="Mock summary",
        count_initial_tokens=True,
    )

    json_str = memory.to_string()

    new_memory = ChatSummaryMemoryBuffer.from_string(json_str)

    assert len(new_memory.get()) == 1
    assert new_memory.token_limit == 5
    assert new_memory.summarize_prompt == "Mock summary"
    assert new_memory.count_initial_tokens
    # The user needs to set the llm manually when loading (and it needs to match the tokenizer_fn)
    assert new_memory.llm is None
    new_memory.llm = summarizer_llm


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_dict_save_load(summarizer_llm) -> None:
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=summarizer_llm,
        chat_history=[USER_CHAT_MESSAGE],
        token_limit=5,
        summarize_prompt="Mock summary",
        count_initial_tokens=True,
    )

    json_dict = memory.to_dict()

    new_memory = ChatSummaryMemoryBuffer.from_dict(json_dict)

    assert len(new_memory.get()) == 1
    assert new_memory.token_limit == 5
    assert new_memory.summarize_prompt == "Mock summary"
    assert new_memory.count_initial_tokens
    # The user needs to set the llm manually when loading (and it needs to match the tokenizer_fn)
    assert new_memory.llm is None
    new_memory.llm = summarizer_llm


@pytest.mark.skipif(not openai_installed, reason="OpenAI not installed")
def test_pickle() -> None:
    """Unpickleable tiktoken tokenizer should be circumvented when pickling."""
    memory = ChatSummaryMemoryBuffer.from_defaults()
    bytes_ = pickle.dumps(memory)
    assert isinstance(pickle.loads(bytes_), ChatSummaryMemoryBuffer)
