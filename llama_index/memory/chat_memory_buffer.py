from pydantic import Field
from typing import List, Callable

from llama_index.llms.base import ChatMessage
from llama_index.memory.types import BaseMemory
from llama_index.utils import GlobalsHelper


class ChatMemoryBuffer(BaseMemory):
    """Simple buffer for storing chat history."""

    chat_history: List[ChatMessage] = Field(default_factory=list)
    token_limit: int = 3000
    tokenizer_fn: Callable[[str], List] = Field(default=GlobalsHelper.tokenizer)

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

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        self.chat_history.append(message)
