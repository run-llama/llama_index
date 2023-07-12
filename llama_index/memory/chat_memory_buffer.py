from pydantic import Field
from typing import Callable, List, Optional

from llama_index.llms.base import ChatMessage, LLM
from llama_index.memory.types import BaseMemory
from llama_index.utils import GlobalsHelper

DEFUALT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


class ChatMemoryBuffer(BaseMemory):
    """Simple buffer for storing chat history."""

    token_limit: int
    tokenizer_fn: Callable[[str], List]
    chat_history: List[ChatMessage] = Field(default_factory=list)

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
    ) -> "ChatMemoryBuffer":
        """Create a chat memory buffer from an LLM."""
        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFUALT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or GlobalsHelper().tokenizer,
            chat_history=chat_history or [],
        )

    def get(self) -> List[ChatMessage]:
        """Get chat history."""
        message_count = len(self.chat_history)
        message_str = " ".join(
            [str(m.content) for m in self.chat_history[-message_count:]]
        )
        token_count = len(self.tokenizer_fn(message_str))

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            message_str = " ".join(
                [str(m.content) for m in self.chat_history[-message_count:]]
            )
            token_count = len(self.tokenizer_fn(message_str))

        # catch one message longer than token limit
        if token_count > self.token_limit:
            return []

        return self.chat_history[:message_count]

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        return self.chat_history

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        self.chat_history.append(message)

    def reset(self) -> None:
        """Reset chat history."""
        return self.chat_history.clear()
