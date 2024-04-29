from typing import Any, Callable, Dict, List, Optional

from pydantic.fields import PrivateAttr

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, root_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer

DEFAULT_TOKEN_LIMIT_FULL_TEXT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT_FULL_TEXT = 2000
SUMMARIZE_PROMPT = "The following is a conversation between the user and assistant. Write a concise summary about the contents of this conversation."


class ChatSummaryMemoryBuffer(BaseMemory):
    """Buffer for storing chat history that uses the full text for the latest
    {interaction_limit_full_text} chat interactions or {token_limit_full_text} tokens.

    All older messages are iteratively summarized using the {summarizer_llm} provided, with
    the max number of tokens defined by the {summarizer_llm}

    This approach is useful to retain long chat history, while limiting the token count
    and latency of each request to the LLM.
    """

    token_limit_full_text: int
    count_initial_tokens: bool = False
    summarizer_llm: Optional[LLM] = None
    summarize_prompt: Optional[str] = None
    tokenizer_fn: Callable[[str], List] = Field(
        # NOTE: mypy does not handle the typing here well, hence the cast
        default_factory=get_tokenizer,
        exclude=True,
    )

    chat_store: BaseChatStore = Field(default_factory=SimpleChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)

    _token_count: int = PrivateAttr(default=0)

    @root_validator(pre=True)
    def validate_memory(cls, values: dict) -> dict:
        # Validate token limits
        token_limit_full_text = values.get("token_limit_full_text", -1)
        if token_limit_full_text < 1:
            raise ValueError(
                "Token limit for full-text messages must be set and greater than 0."
            )

        # Validate tokenizer -- this avoids errors when loading from json/dict
        tokenizer_fn = values.get("tokenizer_fn", None)
        if tokenizer_fn is None:
            values["tokenizer_fn"] = get_tokenizer()

        return values

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        summarizer_llm: Optional[LLM] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit_full_text: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        summarize_prompt: Optional[str] = None,
        count_initial_tokens: bool = False,
    ) -> "ChatSummaryMemoryBuffer":
        """Create a chat memory buffer from an LLM
        and an initial list of chat history messages.
        """
        if summarizer_llm is not None:
            context_window = summarizer_llm.metadata.context_window
            token_limit_full_text = token_limit_full_text or int(
                context_window * DEFAULT_TOKEN_LIMIT_FULL_TEXT_RATIO
            )
        elif token_limit_full_text is None:
            token_limit_full_text = DEFAULT_TOKEN_LIMIT_FULL_TEXT

        chat_store = chat_store or SimpleChatStore()
        chat_history = chat_history or []
        chat_store.set_messages(chat_store_key, chat_history)

        summarize_prompt = summarize_prompt or SUMMARIZE_PROMPT
        return cls(
            summarizer_llm=summarizer_llm,
            token_limit_full_text=token_limit_full_text,
            # TODO: Check if we can get the tokenizer from the llm
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            summarize_prompt=summarize_prompt,
            chat_store=chat_store,
            chat_store_key=chat_store_key,
            count_initial_tokens=count_initial_tokens,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ChatSummaryMemoryBuffer"

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    def to_dict(self, **kwargs: Any) -> dict:
        """Convert memory to dict."""
        return self.dict()

    @classmethod
    def from_string(cls, json_str: str) -> "ChatMemoryBuffer":
        raise NotImplementedError("This is not yet supported.")

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], **kwargs: Any
    ) -> "ChatSummaryMemoryBuffer":
        raise NotImplementedError("This is not yet supported.")

    def get(self, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        chat_history = self.get_all()
        if len(chat_history) == 0:
            return []

        # Give the user the choice whether to count the system prompt or not
        if self.count_initial_tokens:
            if initial_token_count > self.token_limit_full_text:
                raise ValueError("Initial token count exceeds token limit")
            self._token_count = initial_token_count

        (
            chat_history_full_text,
            chat_history_to_be_summarized,
        ) = self._split_messages_summary_or_full_text(chat_history)

        if self.summarizer_llm is None or len(chat_history_to_be_summarized) == 0:
            # Simply remove the message that don't fit the buffer anymore
            updated_history = chat_history_full_text
        else:
            updated_history = [
                self._summarize_oldest_chat_history(chat_history_to_be_summarized),
                *chat_history_full_text,
            ]

        self.reset()
        self._token_count = 0
        self.set(updated_history)

        return updated_history

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        return self.chat_store.get_messages(self.chat_store_key)

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        # ensure everything is serialized
        self.chat_store.add_message(self.chat_store_key, message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.chat_store.set_messages(self.chat_store_key, messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_store.delete_messages(self.chat_store_key)

    def get_token_count(self) -> int:
        """Returns the token count of the memory buffer (excluding the last assistant response)."""
        return self._token_count

    def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        """Get token count for list of messages."""
        if len(messages) <= 0:
            return 0

        msg_str = " ".join(str(m.content) for m in messages)
        return len(self.tokenizer_fn(msg_str))

    def _split_messages_summary_or_full_text(
        self, chat_history: List[ChatMessage]
    ) -> (List[ChatMessage], List[ChatMessage]):
        """Determine which messages will be included as full text,
        and which will have to be summarized by the summarizer_llm.
        """
        chat_history_full_text = []
        message_count = len(chat_history)
        while (
            message_count > 0
            and self._token_count + self._token_count_for_messages([chat_history[-1]])
            <= self.token_limit_full_text
        ):
            # traverse the history in reverse order, when token limit is about to be
            # exceeded, we stop, so remaining messages are summarized
            self._token_count += self._token_count_for_messages([chat_history[-1]])
            chat_history_full_text.insert(0, chat_history.pop())
            message_count -= 1

        chat_history_to_be_summarized = chat_history.copy()
        return chat_history_full_text, chat_history_to_be_summarized

    def _summarize_oldest_chat_history(
        self, chat_history_to_be_summarized: List[ChatMessage]
    ) -> ChatMessage:
        """Use the summarizer_llm to summarize the messages that do not fit into the
        buffer.
        """
        # Only summarize if there is new information to be summarized
        if (
            len(chat_history_to_be_summarized) == 1
            and chat_history_to_be_summarized[0].role == MessageRole.SYSTEM
        ):
            return chat_history_to_be_summarized[0]

        summarize_prompt = ChatMessage(
            role=MessageRole.SYSTEM,
            content=self._get_prompt_to_summarize(chat_history_to_be_summarized),
        )
        # TODO: Maybe it is better to pass a list of history to llm
        return ChatMessage(
            role=MessageRole.SYSTEM,
            content=self.summarizer_llm.chat([summarize_prompt]).message.content,
        )

    def _get_prompt_to_summarize(
        self, chat_history_to_be_summarized: List[ChatMessage]
    ):
        """Ask the LLM to summarize the chat history so far."""
        # TODO: This probably works better when question/answers are considered together.
        prompt = '"Transcript so far: '
        for msg in chat_history_to_be_summarized:
            prompt += msg.role + ": "
            prompt += msg.content + "\n\n"
        prompt += '"\n\n'
        prompt += self.summarize_prompt
        return prompt
