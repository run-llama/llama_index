import json
import logging
from typing import Any, Callable, Dict, List, Tuple, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
    model_validator,
    field_serializer,
    SerializeAsAny,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 2000
SUMMARIZE_PROMPT = "The following is a conversation between the user and assistant. Write a concise summary about the contents of this conversation."

logger = logging.getLogger(__name__)


# TODO: Add option for last N user/assistant history interactions instead of token limit
class ChatSummaryMemoryBuffer(BaseMemory):
    """Buffer for storing chat history that uses the full text for the latest
    {token_limit}.

    All older messages are iteratively summarized using the {llm} provided, with
    the max number of tokens defined by the {llm}.

    User can specify whether initial tokens (usually a system prompt)
    should be counted as part of the {token_limit}
    using the parameter {count_initial_tokens}.

    This buffer is useful to retain the most important information from a
    long chat history, while limiting the token count and latency
    of each request to the LLM.
    """

    token_limit: int
    count_initial_tokens: bool = False
    llm: Optional[SerializeAsAny[LLM]] = None
    summarize_prompt: Optional[str] = None
    tokenizer_fn: Callable[[str], List] = Field(
        # NOTE: mypy does not handle the typing here well, hence the cast
        default_factory=get_tokenizer,
        exclude=True,
    )

    chat_store: SerializeAsAny[BaseChatStore] = Field(default_factory=SimpleChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)

    _token_count: int = PrivateAttr(default=0)

    @field_serializer("chat_store")
    def serialize_courses_in_order(chat_store: BaseChatStore):
        res = chat_store.model_dump()
        res.update({"class_name": chat_store.class_name()})
        return res

    @model_validator(mode="before")
    @classmethod
    def validate_memory(cls, values: dict) -> dict:
        """Validate the memory."""
        # Validate token limits
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
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
        llm: Optional[LLM] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        summarize_prompt: Optional[str] = None,
        count_initial_tokens: bool = False,
    ) -> "ChatSummaryMemoryBuffer":
        """Create a chat memory buffer from an LLM
        and an initial list of chat history messages.
        """
        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFAULT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        chat_store = chat_store or SimpleChatStore()

        if chat_history is not None:
            chat_store.set_messages(chat_store_key, chat_history)

        summarize_prompt = summarize_prompt or SUMMARIZE_PROMPT
        return cls(
            llm=llm,
            token_limit=token_limit,
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
    def from_string(cls, json_str: str, **kwargs: Any) -> "ChatSummaryMemoryBuffer":
        """Create a chat memory buffer from a string."""
        dict_obj = json.loads(json_str)
        return cls.from_dict(dict_obj, **kwargs)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], **kwargs: Any
    ) -> "ChatSummaryMemoryBuffer":
        from llama_index.core.storage.chat_store.loading import load_chat_store

        # NOTE: this handles backwards compatibility with the old chat history
        if "chat_history" in data:
            chat_history = data.pop("chat_history")
            chat_store = SimpleChatStore(store={DEFAULT_CHAT_STORE_KEY: chat_history})
            data["chat_store"] = chat_store
        elif "chat_store" in data:
            chat_store = data.pop("chat_store")
            chat_store = load_chat_store(chat_store)
            data["chat_store"] = chat_store

        # NOTE: The llm will have to be set manually in kwargs
        if "llm" in data:
            data.pop("llm")

        return cls(**data, **kwargs)

    def get(self, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        chat_history = self.get_all()
        if len(chat_history) == 0:
            return []

        # Give the user the choice whether to count the system prompt or not
        if self.count_initial_tokens:
            if initial_token_count > self.token_limit:
                raise ValueError("Initial token count exceeds token limit")
            self._token_count = initial_token_count

        (
            chat_history_full_text,
            chat_history_to_be_summarized,
        ) = self._split_messages_summary_or_full_text(chat_history)

        if self.llm is None or len(chat_history_to_be_summarized) == 0:
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
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """Determine which messages will be included as full text,
        and which will have to be summarized by the llm.
        """
        chat_history_full_text = []
        message_count = len(chat_history)
        while (
            message_count > 0
            and self.get_token_count()
            + self._token_count_for_messages([chat_history[-1]])
            <= self.token_limit
        ):
            # traverse the history in reverse order, when token limit is about to be
            # exceeded, we stop, so remaining messages are summarized
            self._token_count += self._token_count_for_messages([chat_history[-1]])
            chat_history_full_text.insert(0, chat_history.pop())
            message_count -= 1

        chat_history_to_be_summarized = chat_history.copy()
        self._handle_assistant_and_tool_messages(
            chat_history_full_text, chat_history_to_be_summarized
        )

        return chat_history_full_text, chat_history_to_be_summarized

    def _summarize_oldest_chat_history(
        self, chat_history_to_be_summarized: List[ChatMessage]
    ) -> ChatMessage:
        """Use the llm to summarize the messages that do not fit into the
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
            content=self.llm.chat([summarize_prompt]).message.content,
        )

    def _get_prompt_to_summarize(
        self, chat_history_to_be_summarized: List[ChatMessage]
    ):
        """Ask the LLM to summarize the chat history so far."""
        # TODO: This probably works better when question/answers are considered together.
        prompt = '"Transcript so far: '
        for msg in chat_history_to_be_summarized:
            prompt += msg.role + ": "
            if msg.content:
                prompt += msg.content + "\n\n"
            else:
                prompt += (
                    "\n".join(
                        [
                            f"Calling a function: {call!s}"
                            for call in msg.additional_kwargs.get("tool_calls", [])
                        ]
                    )
                    + "\n\n"
                )
        prompt += '"\n\n'
        prompt += self.summarize_prompt
        return prompt

    def _handle_assistant_and_tool_messages(
        self,
        chat_history_full_text: List[ChatMessage],
        chat_history_to_be_summarized: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """To avoid breaking API's, we need to ensure the following.

        - the first message cannot be ASSISTANT
        - ASSISTANT/TOOL should be considered in pairs
        Therefore, we switch messages to summarized list until the first message is
        not an ASSISTANT or TOOL message.
        """
        while chat_history_full_text and chat_history_full_text[0].role in (
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ):
            chat_history_to_be_summarized.append(chat_history_full_text.pop(0))
