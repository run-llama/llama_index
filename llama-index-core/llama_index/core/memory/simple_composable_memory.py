from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.memory.types import (
    BaseMemory,
)
from llama_index.core.memory import ChatMemoryBuffer

DEFAULT_INTRO_HISTORY_MESSAGE = "Below are a set of relevant dialogues retrieved from potentially several memory sources:"
DEFAULT_OUTRO_HISTORY_MESSAGE = "This is the end of the retrieved message dialogues."


class SimpleComposableMemory(BaseMemory):
    """A simple composition of potentially several memory sources.

    This composable memory considers one of the memory sources as the main
    one and the others as secondary. The secondary memory sources get added to
    the chat history only in either the system prompt or to the first user
    message within the chat history.

    Args:
        primary_memory: (BaseMemory) The main memory buffer for agent.
        secondary_memory_sources: (List(BaseMemory)) Secondary memory sources.
            Retrieved messages from these sources get added to the system prompt message.
    """

    primary_memory: SerializeAsAny[BaseMemory] = Field(
        description="Primary memory source for chat agent.",
    )
    secondary_memory_sources: List[SerializeAsAny[BaseMemory]] = Field(
        default_factory=list, description="Secondary memory sources."
    )

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SimpleComposableMemory"

    @classmethod
    def from_defaults(
        cls,
        primary_memory: Optional[BaseMemory] = None,
        secondary_memory_sources: Optional[List[BaseMemory]] = None,
        **kwargs: Any,
    ) -> "SimpleComposableMemory":
        """Create a simple composable memory from an LLM."""
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        primary_memory = primary_memory or ChatMemoryBuffer.from_defaults()
        secondary_memory_sources = secondary_memory_sources or []

        return cls(
            primary_memory=primary_memory,
            secondary_memory_sources=secondary_memory_sources,
        )

    def _format_secondary_messages(
        self, secondary_chat_histories: List[List[ChatMessage]]
    ) -> str:
        """Formats retrieved historical messages into a single string."""
        # TODO: use PromptTemplate for this
        formatted_history = "\n\n" + DEFAULT_INTRO_HISTORY_MESSAGE + "\n"
        for ix, chat_history in enumerate(secondary_chat_histories):
            formatted_history += (
                f"\n=====Relevant messages from memory source {ix + 1}=====\n\n"
            )
            for m in chat_history:
                formatted_history += f"\t{m.role.upper()}: {m.content}\n"
            formatted_history += (
                f"\n=====End of relevant messages from memory source {ix + 1}======\n\n"
            )

        formatted_history += DEFAULT_OUTRO_HISTORY_MESSAGE
        return formatted_history

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        return self._compose_message_histories(input, **kwargs)

    def _compose_message_histories(
        self, input: Optional[str] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Get chat history."""
        # get from primary
        messages = self.primary_memory.get(input=input, **kwargs)

        # get from secondary
        # TODO: remove any repeated messages in secondary and primary memory
        secondary_histories = []
        for mem in self.secondary_memory_sources:
            secondary_history = mem.get(input, **kwargs)
            secondary_history = [m for m in secondary_history if m not in messages]

            if len(secondary_history) > 0:
                secondary_histories.append(secondary_history)

        # format secondary memory
        if len(secondary_histories) > 0:
            single_secondary_memory_str = self._format_secondary_messages(
                secondary_histories
            )

            # add single_secondary_memory_str to chat_history
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
                assert messages[0].content is not None
                system_message = messages[0].content.split(
                    DEFAULT_INTRO_HISTORY_MESSAGE
                )[0]
                messages[0] = ChatMessage(
                    content=system_message.strip() + single_secondary_memory_str,
                    role=MessageRole.SYSTEM,
                )
            else:
                messages.insert(
                    0,
                    ChatMessage(
                        content="You are a helpful assistant."
                        + single_secondary_memory_str,
                        role=MessageRole.SYSTEM,
                    ),
                )
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history.

        Uses primary memory get_all only.
        """
        return self.primary_memory.get_all()

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        self.primary_memory.put(message)
        for mem in self.secondary_memory_sources:
            mem.put(message)

    async def aput(self, message: ChatMessage) -> None:
        """Put chat history."""
        await self.primary_memory.aput(message)
        for mem in self.secondary_memory_sources:
            await mem.aput(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.primary_memory.set(messages)
        for mem in self.secondary_memory_sources:
            # finalize task often sets, but secondary memory is meant for
            # long-term memory rather than main chat memory buffer
            # so use put_messages instead
            mem.put_messages(messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.primary_memory.reset()
        for mem in self.secondary_memory_sources:
            mem.reset()
