import json
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, model_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import (
    DEFAULT_CHAT_STORE_KEY,
    BaseChatStoreMemory,
)
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


class ChatMemoryBuffer(BaseChatStoreMemory):
    """Simple buffer for storing chat history."""

    token_limit: int
    tokenizer_fn: Callable[[str], List] = Field(
        default_factory=get_tokenizer,
        exclude=True,
    )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ChatMemoryBuffer"

    @model_validator(mode="before")
    @classmethod
    def validate_memory(cls, values: dict) -> dict:
        # Validate token limit
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")

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
        **kwargs: Any,
    ) -> "ChatMemoryBuffer":
        """Create a chat memory buffer from an LLM."""
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFAULT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        if chat_history is not None:
            chat_store = chat_store or SimpleChatStore()
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            chat_store=chat_store or SimpleChatStore(),
            chat_store_key=chat_store_key,
        )

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    @classmethod
    def from_string(cls, json_str: str) -> "ChatMemoryBuffer":
        """Create a chat memory buffer from a string."""
        dict_obj = json.loads(json_str)
        print(f"dict_obj: {dict_obj}", flush=True)
        return cls.from_dict(dict_obj)

    def to_dict(self, **kwargs: Any) -> dict:
        """Convert memory to dict."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "ChatMemoryBuffer":
        from llama_index.core.storage.chat_store.loading import load_chat_store

        # NOTE: this handles backwards compatibility with the old chat history
        if "chat_history" in data:
            chat_history = data.pop("chat_history")
            simple_store = SimpleChatStore(store={DEFAULT_CHAT_STORE_KEY: chat_history})
            data["chat_store"] = simple_store
        elif "chat_store" in data:
            chat_store_dict = data.pop("chat_store")
            chat_store = load_chat_store(chat_store_dict)
            data["chat_store"] = chat_store

        return cls(**data)

    def get(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> List[ChatMessage]:
        """Get chat history."""
        chat_history = self.get_all()

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        message_count = len(chat_history)

        cur_messages = chat_history[-message_count:]
        token_count = self._token_count_for_messages(cur_messages) + initial_token_count

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            if chat_history[-message_count].role == MessageRole.TOOL:
                # all tool messages should be preceded by an assistant message
                # if we remove a tool message, we need to remove the assistant message too
                message_count -= 1

            if chat_history[-message_count].role == MessageRole.ASSISTANT:
                # we cannot have an assistant message at the start of the chat history
                # if after removal of the first, we have an assistant message,
                # we need to remove the assistant message too
                message_count -= 1

            cur_messages = chat_history[-message_count:]
            token_count = (
                self._token_count_for_messages(cur_messages) + initial_token_count
            )

        # catch one message longer than token limit
        if token_count > self.token_limit or message_count <= 0:
            return []

        return chat_history[-message_count:]

    def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        if len(messages) <= 0:
            return 0

        msg_str = " ".join(str(m.content) for m in messages)
        return len(self.tokenizer_fn(msg_str))
