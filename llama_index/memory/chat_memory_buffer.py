from pydantic import Field
from typing import Callable, List, Optional

from llama_index.llms.base import ChatMessage, LLM
from llama_index.memory.types import BaseMemory
from llama_index.utils import GlobalsHelper

DEFUALT_TOKEN_LIMIT_RATIO = 0.75


class ChatMemoryBuffer(BaseMemory):
    """Simple buffer for storing chat history."""

    token_limit: int
    chat_history: List[ChatMessage] = Field(default_factory=list)
    tokenizer_fn: Callable[[str], List] = Field(default=GlobalsHelper.tokenizer)

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
    ) -> "ChatMemoryBuffer":
        """Create a chat memory buffer from an LLM."""
        context_window = llm.metadata.context_window
        token_limit = token_limit or int(context_window * DEFUALT_TOKEN_LIMIT_RATIO)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or GlobalsHelper.tokenizer,
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

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        self.chat_history.append(message)
