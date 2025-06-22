import asyncio
import uuid
from abc import abstractmethod
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Generic,
    cast,
)

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import (
    ChatMessage,
    ContentBlock,
    TextBlock,
    AudioBlock,
    ImageBlock,
    DocumentBlock,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
)
from llama_index.core.memory.types import BaseMemory
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore, MessageStatus
from llama_index.core.utils import get_tokenizer

# Define type variable for memory block content
T = TypeVar("T", str, List[ContentBlock], List[ChatMessage])

DEFAULT_TOKEN_LIMIT = 30000
DEFAULT_FLUSH_SIZE = int(DEFAULT_TOKEN_LIMIT * 0.1)
DEFAULT_MEMORY_BLOCKS_TEMPLATE = RichPromptTemplate(
    """
<memory>
{% for (block_name, block_content) in memory_blocks %}
<{{ block_name }}>
  {% for block in block_content %}
    {% if block.block_type == "text" %}
{{ block.text }}
    {% elif block.block_type == "image" %}
      {% if block.url %}
        {{ (block.url | string) | image }}
      {% elif block.path %}
        {{ (block.path | string) | image }}
      {% endif %}
    {% elif block.block_type == "audio" %}
      {% if block.url %}
        {{ (block.url | string) | audio }}
      {% elif block.path %}
        {{ (block.path | string) | audio }}
      {% endif %}
    {% endif %}
  {% endfor %}
</{{ block_name }}>
{% endfor %}
</memory>
"""
)


class InsertMethod(Enum):
    SYSTEM = "system"
    USER = "user"


def generate_chat_store_key() -> str:
    """Generate a unique chat store key."""
    return str(uuid.uuid4())


def get_default_chat_store() -> SQLAlchemyChatStore:
    """Get the default chat store."""
    return SQLAlchemyChatStore(table_name="llama_index_memory")


class BaseMemoryBlock(BaseModel, Generic[T]):
    """
    A base class for memory blocks.

    Subclasses must implement the `aget` and `aput` methods.
    Optionally, subclasses can implement the `atruncate` method, which is used to reduce the size of the memory block.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="The name/identifier of the memory block.")
    description: Optional[str] = Field(
        default=None, description="A description of the memory block."
    )
    priority: int = Field(
        default=0,
        description="Priority of this memory block (0 = never truncate, 1 = highest priority, etc.).",
    )
    accept_short_term_memory: bool = Field(
        default=True,
        description="Whether to accept puts from messages ejected from the short-term memory.",
    )

    @abstractmethod
    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> T:
        """Pull the memory block (async)."""

    async def aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> T:
        """
        Pull the memory block (async).

        Returns:
            T: The memory block content. One of:
            - str: A simple text string to be inserted into the template.
            - List[ContentBlock]: A list of content blocks to be inserted into the template.
            - List[ChatMessage]: A list of chat messages to be directly appended to the chat history.

        """
        return await self._aget(messages, **block_kwargs)

    @abstractmethod
    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Push to the memory block (async)."""

    async def aput(
        self,
        messages: List[ChatMessage],
        from_short_term_memory: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        """Push to the memory block (async)."""
        if from_short_term_memory and not self.accept_short_term_memory:
            return

        if session_id is not None:
            for message in messages:
                message.additional_kwargs["session_id"] = session_id

        await self._aput(messages)

    async def atruncate(self, content: T, tokens_to_truncate: int) -> Optional[T]:
        """
        Truncate the memory block content to the given token limit.

        By default, truncation will remove the entire block content.

        Args:
            content:
                The content of type T, depending on what the memory block returns.
            tokens_to_truncate:
                The number of tokens requested to truncate the content by.
                Blocks may or may not truncate to the exact number of tokens requested, but it
                can be used as a hint for the block to truncate.

        Returns:
            The truncated content of type T, or None if the content is completely truncated.

        """
        return None


class Memory(BaseMemory):
    """
    A memory module that waterfalls into memory blocks.

    Works by orchestrating around
    - a FIFO queue of messages
    - a list of memory blocks
    - various parameters (pressure size, token limit, etc.)

    When the FIFO queue reaches the token limit, the oldest messages within the pressure size are ejected from the FIFO queue.
    The messages are then processed by each memory block.

    When pulling messages from this memory, the memory blocks are processed in order, and the messages are injected into the system message or the latest user message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_limit: int = Field(
        default=DEFAULT_TOKEN_LIMIT,
        description="The overall token limit of the memory.",
    )
    token_flush_size: int = Field(
        default=DEFAULT_FLUSH_SIZE,
        description="The token size to use for flushing the FIFO queue.",
    )
    chat_history_token_ratio: float = Field(
        default=0.7,
        description="Minimum percentage ratio of total token limit reserved for chat history.",
    )
    memory_blocks: List[BaseMemoryBlock] = Field(
        default_factory=list,
        description="The list of memory blocks to use.",
    )
    memory_blocks_template: RichPromptTemplate = Field(
        default=DEFAULT_MEMORY_BLOCKS_TEMPLATE,
        description="The template to use for formatting the memory blocks.",
    )
    insert_method: InsertMethod = Field(
        default=InsertMethod.SYSTEM,
        description="Whether to inject memory blocks into a system message or into the latest user message.",
    )
    image_token_size_estimate: int = Field(
        default=256,
        description="The token size estimate for images.",
    )
    audio_token_size_estimate: int = Field(
        default=256,
        description="The token size estimate for audio.",
    )
    tokenizer_fn: Callable[[str], List] = Field(
        default_factory=get_tokenizer,
        exclude=True,
        description="The tokenizer function to use for token counting.",
    )
    sql_store: SQLAlchemyChatStore = Field(
        default_factory=get_default_chat_store,
        exclude=True,
        description="The chat store to use for storing messages.",
    )
    session_id: str = Field(
        default_factory=generate_chat_store_key,
        description="The key to use for storing messages in the chat store.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "Memory"

    @model_validator(mode="before")
    @classmethod
    def validate_memory(cls, values: dict) -> dict:
        # Validate token limit
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")

        tokenizer_fn = values.get("tokenizer_fn")
        if tokenizer_fn is None:
            values["tokenizer_fn"] = get_tokenizer()

        if values.get("token_flush_size", -1) < 1:
            values["token_flush_size"] = int(token_limit * 0.1)
        elif values.get("token_flush_size", -1) > token_limit:
            values["token_flush_size"] = int(token_limit * 0.1)

        # validate all blocks have unique names
        block_names = [block.name for block in values.get("memory_blocks", [])]
        if len(block_names) != len(set(block_names)):
            raise ValueError("All memory blocks must have unique names.")

        return values

    @classmethod
    def from_defaults(  # type: ignore[override]
        cls,
        session_id: Optional[str] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        token_limit: int = DEFAULT_TOKEN_LIMIT,
        memory_blocks: Optional[List[BaseMemoryBlock[Any]]] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        chat_history_token_ratio: float = 0.7,
        token_flush_size: int = DEFAULT_FLUSH_SIZE,
        memory_blocks_template: RichPromptTemplate = DEFAULT_MEMORY_BLOCKS_TEMPLATE,
        insert_method: InsertMethod = InsertMethod.SYSTEM,
        image_token_size_estimate: int = 256,
        audio_token_size_estimate: int = 256,
        # SQLAlchemyChatStore parameters
        table_name: str = "llama_index_memory",
        async_database_uri: Optional[str] = None,
        async_engine: Optional[AsyncEngine] = None,
    ) -> "Memory":
        """Initialize Memory."""
        session_id = session_id or generate_chat_store_key()

        # If not using the SQLAlchemyChatStore, provide an error
        sql_store = SQLAlchemyChatStore(
            table_name=table_name,
            async_database_uri=async_database_uri,
            async_engine=async_engine,
        )

        if chat_history is not None:
            asyncio_run(sql_store.set_messages(session_id, chat_history))

        if token_flush_size > token_limit:
            token_flush_size = int(token_limit * 0.7)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            sql_store=sql_store,
            session_id=session_id,
            memory_blocks=memory_blocks or [],
            chat_history_token_ratio=chat_history_token_ratio,
            token_flush_size=token_flush_size,
            memory_blocks_template=memory_blocks_template,
            insert_method=insert_method,
            image_token_size_estimate=image_token_size_estimate,
            audio_token_size_estimate=audio_token_size_estimate,
        )

    def _estimate_token_count(
        self,
        message_or_blocks: Union[
            str, ChatMessage, List[ChatMessage], List[ContentBlock]
        ],
    ) -> int:
        """Estimate token count for a message."""
        token_count = 0

        # Normalize the input to a list of ContentBlocks
        if isinstance(message_or_blocks, ChatMessage):
            blocks = message_or_blocks.blocks

            # Estimate the token count for the additional kwargs
            if message_or_blocks.additional_kwargs:
                token_count += len(
                    self.tokenizer_fn(str(message_or_blocks.additional_kwargs))
                )
        elif isinstance(message_or_blocks, List):
            # Type narrow the list
            messages: List[ChatMessage] = []
            content_blocks: List[
                Union[TextBlock, ImageBlock, AudioBlock, DocumentBlock]
            ] = []

            if all(isinstance(item, ChatMessage) for item in message_or_blocks):
                messages = cast(List[ChatMessage], message_or_blocks)

                blocks = []
                for msg in messages:
                    blocks.extend(msg.blocks)

                # Estimate the token count for the additional kwargs
                token_count += sum(
                    len(self.tokenizer_fn(str(msg.additional_kwargs)))
                    for msg in messages
                    if msg.additional_kwargs
                )
            elif all(
                isinstance(item, (TextBlock, ImageBlock, AudioBlock, DocumentBlock))
                for item in message_or_blocks
            ):
                content_blocks = cast(
                    List[Union[TextBlock, ImageBlock, AudioBlock, DocumentBlock]],
                    message_or_blocks,
                )
                blocks = content_blocks
            else:
                raise ValueError(f"Invalid message type: {type(message_or_blocks)}")
        elif isinstance(message_or_blocks, str):
            blocks = [TextBlock(text=message_or_blocks)]
        else:
            raise ValueError(f"Invalid message type: {type(message_or_blocks)}")

        # Estimate the token count for each block
        for block in blocks:
            if isinstance(block, TextBlock):
                token_count += len(self.tokenizer_fn(block.text))
            elif isinstance(block, ImageBlock):
                token_count += self.image_token_size_estimate
            elif isinstance(block, AudioBlock):
                token_count += self.audio_token_size_estimate

        return token_count

    async def _get_memory_blocks_content(
        self,
        chat_history: List[ChatMessage],
        input: Optional[Union[str, ChatMessage]] = None,
        **block_kwargs: Any,
    ) -> Dict[str, Any]:
        """Get content from memory blocks in priority order."""
        content_per_memory_block: Dict[str, Any] = {}

        block_input = chat_history
        if isinstance(input, str):
            block_input = [*chat_history, ChatMessage(role="user", content=input)]

        # Process memory blocks in priority order
        for memory_block in sorted(self.memory_blocks, key=lambda x: -x.priority):
            content = await memory_block.aget(
                block_input, session_id=self.session_id, **block_kwargs
            )

            # Handle different return types from memory blocks
            if content and isinstance(content, list):
                # Memory block returned content blocks
                content_per_memory_block[memory_block.name] = content
            elif content and isinstance(content, str):
                # Memory block returned a string
                content_per_memory_block[memory_block.name] = content
            elif not content:
                continue
            else:
                raise ValueError(
                    f"Invalid content type received from memory block {memory_block.name}: {type(content)}"
                )

        return content_per_memory_block

    async def _truncate_memory_blocks(
        self,
        content_per_memory_block: Dict[str, Any],
        memory_blocks_tokens: int,
        chat_history_tokens: int,
    ) -> Dict[str, Any]:
        """Truncate memory blocks if total token count exceeds limit."""
        if memory_blocks_tokens + chat_history_tokens <= self.token_limit:
            return content_per_memory_block

        tokens_to_truncate = (
            memory_blocks_tokens + chat_history_tokens - self.token_limit
        )
        truncated_content = content_per_memory_block.copy()

        # Truncate memory blocks based on priority
        for memory_block in sorted(
            self.memory_blocks, key=lambda x: x.priority
        ):  # Lower priority first
            # Skip memory blocks with priority 0, they should never be truncated
            if memory_block.priority == 0:
                continue

            if tokens_to_truncate <= 0:
                break

            # Truncate content and measure tokens saved
            content = truncated_content.get(memory_block.name, [])

            truncated_block_content = await memory_block.atruncate(
                content, tokens_to_truncate
            )

            # Calculate tokens saved
            original_tokens = self._estimate_token_count(content)

            if truncated_block_content is None:
                new_tokens = 0
            else:
                new_tokens = self._estimate_token_count(truncated_block_content)

            tokens_saved = original_tokens - new_tokens
            tokens_to_truncate -= tokens_saved

            # Update the content blocks
            if truncated_block_content is None:
                truncated_content[memory_block.name] = []
            else:
                truncated_content[memory_block.name] = truncated_block_content

        # handle case where we still have tokens to truncate
        # just remove the blocks starting from the least priority
        for memory_block in sorted(self.memory_blocks, key=lambda x: x.priority):
            if memory_block.priority == 0:
                continue

            if tokens_to_truncate <= 0:
                break

            # Truncate content and measure tokens saved
            content = truncated_content.pop(memory_block.name)
            tokens_to_truncate -= self._estimate_token_count(content)

        return truncated_content

    async def _format_memory_blocks(
        self, content_per_memory_block: Dict[str, Any]
    ) -> Tuple[List[Tuple[str, List[ContentBlock]]], List[ChatMessage]]:
        """Format memory blocks content into template data and chat messages."""
        memory_blocks_data: List[Tuple[str, List[ContentBlock]]] = []
        chat_message_data: List[ChatMessage] = []

        for block in self.memory_blocks:
            if block.name in content_per_memory_block:
                content = content_per_memory_block[block.name]

                # Skip empty memory blocks
                if not content:
                    continue

                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], ChatMessage)
                ):
                    chat_message_data.extend(content)
                elif isinstance(content, str):
                    memory_blocks_data.append((block.name, [TextBlock(text=content)]))
                else:
                    memory_blocks_data.append((block.name, content))

        return memory_blocks_data, chat_message_data

    def _insert_memory_content(
        self,
        chat_history: List[ChatMessage],
        memory_content: List[ContentBlock],
        chat_message_data: List[ChatMessage],
    ) -> List[ChatMessage]:
        """Insert memory content into chat history based on insert method."""
        result = chat_history.copy()

        # Process chat messages
        if chat_message_data:
            result = [*chat_message_data, *result]

        # Process template-based memory blocks
        if memory_content:
            if self.insert_method == InsertMethod.SYSTEM:
                # Find system message or create a new one
                system_idx = next(
                    (i for i, msg in enumerate(result) if msg.role == "system"), None
                )

                if system_idx is not None:
                    # Update existing system message
                    result[system_idx].blocks = [
                        *memory_content,
                        *result[system_idx].blocks,
                    ]
                else:
                    # Create new system message at the beginning
                    result.insert(0, ChatMessage(role="system", blocks=memory_content))
            elif self.insert_method == InsertMethod.USER:
                # Find the latest user message
                session_idx = next(
                    (i for i, msg in enumerate(reversed(result)) if msg.role == "user"),
                    None,
                )

                if session_idx is not None:
                    # Get actual index (since we enumerated in reverse)
                    actual_idx = len(result) - 1 - session_idx
                    # Update existing user message
                    result[actual_idx].blocks = [
                        *memory_content,
                        *result[actual_idx].blocks,
                    ]
                else:
                    result.append(ChatMessage(role="user", blocks=memory_content))

        return result

    async def aget(
        self, input: Optional[Union[str, ChatMessage]] = None, **block_kwargs: Any
    ) -> List[ChatMessage]:  # type: ignore[override]
        """Get messages with memory blocks included (async)."""
        # Get chat history efficiently
        chat_history = await self.sql_store.get_messages(
            self.session_id, status=MessageStatus.ACTIVE
        )
        chat_history_tokens = sum(
            self._estimate_token_count(message) for message in chat_history
        )

        # Get memory blocks content
        content_per_memory_block = await self._get_memory_blocks_content(
            chat_history, input=input, **block_kwargs
        )

        # Calculate memory blocks tokens
        memory_blocks_tokens = sum(
            self._estimate_token_count(content)
            for content in content_per_memory_block.values()
        )

        # Handle truncation if needed
        truncated_content = await self._truncate_memory_blocks(
            content_per_memory_block, memory_blocks_tokens, chat_history_tokens
        )

        # Format template-based memory blocks
        memory_blocks_data, chat_message_data = await self._format_memory_blocks(
            truncated_content
        )

        # Create messages from template content
        memory_content = []
        if memory_blocks_data:
            memory_block_messages = self.memory_blocks_template.format_messages(
                memory_blocks=memory_blocks_data
            )
            memory_content = (
                memory_block_messages[0].blocks if memory_block_messages else []
            )

        # Insert memory content into chat history
        return self._insert_memory_content(
            chat_history, memory_content, chat_message_data
        )

    async def _manage_queue(self) -> None:
        """
        Manage the FIFO queue.

        This function manages the memory queue using a waterfall approach:
        1. If the queue exceeds the token limit, it removes oldest messages first
        2. Removed messages are archived and passed to memory blocks
        3. It ensures conversation integrity by keeping related messages together
        4. It maintains at least one complete conversation turn
        """
        # Calculate if we need to waterfall
        current_queue = await self.sql_store.get_messages(
            self.session_id, status=MessageStatus.ACTIVE
        )

        # If current queue is empty, return
        if not current_queue:
            return

        tokens_in_current_queue = sum(
            self._estimate_token_count(message) for message in current_queue
        )

        # If we're over the token limit, initiate waterfall
        token_limit = self.token_limit * self.chat_history_token_ratio
        if tokens_in_current_queue > token_limit:
            # Process from oldest to newest, but efficiently with pop() operations
            reversed_queue = current_queue[::-1]  # newest first, oldest last

            # Calculate approximate number of messages to remove
            tokens_to_remove = tokens_in_current_queue - token_limit

            while tokens_to_remove > 0:
                # If only one message left, keep it regardless of token count
                if len(reversed_queue) <= 1:
                    break

                # Collect messages to flush (up to flush size)
                messages_to_flush = []
                flushed_tokens = 0

                # Remove oldest messages (from end of reversed list) until reaching flush size
                while (
                    flushed_tokens < self.token_flush_size
                    and reversed_queue
                    and len(reversed_queue) > 1
                ):
                    message = reversed_queue.pop()
                    messages_to_flush.append(message)
                    flushed_tokens += self._estimate_token_count(message)

                # Ensure we keep at least one message
                if not reversed_queue and messages_to_flush:
                    reversed_queue.append(messages_to_flush.pop())

                # We need to maintain conversation integrity
                # Messages should be removed in complete conversation turns
                chronological_view = reversed_queue[::-1]  # View in chronological order

                # Find the correct conversation boundary
                # We want the first message in our remaining queue to be a user message
                # and the last message to be from assistant or tool
                if chronological_view:
                    # Keep removing messages until first remaining message is from user
                    # This ensures we start with a user message
                    while (
                        chronological_view
                        and chronological_view[0].role != "user"
                        and len(reversed_queue) > 1
                    ):
                        if reversed_queue:
                            messages_to_flush.append(reversed_queue.pop())
                            chronological_view = reversed_queue[::-1]
                        else:
                            break

                    # If we end up with an empty queue, keep at least one full conversation turn
                    if not reversed_queue and messages_to_flush:
                        # Find the most recent complete conversation turn
                        # (user → assistant/tool sequence) in messages_to_flush
                        found_user = False
                        turn_messages: List[ChatMessage] = []

                        # Go through messages_to_flush in reverse (newest first)
                        for msg in reversed(messages_to_flush):
                            if msg.role == "user" and not found_user:
                                found_user = True
                                turn_messages.insert(0, msg)
                            elif found_user:
                                turn_messages.insert(0, msg)
                            else:
                                break

                        # If we found a complete turn, keep it
                        if found_user and turn_messages:
                            # Remove these messages from messages_to_flush
                            for msg in turn_messages:
                                messages_to_flush.remove(msg)
                            # Add them back to the queue
                            reversed_queue = turn_messages[::-1] + reversed_queue

                # Archive the flushed messages
                if messages_to_flush:
                    await self.sql_store.archive_oldest_messages(
                        self.session_id, n=len(messages_to_flush)
                    )

                    # Waterfall the flushed messages to memory blocks
                    await asyncio.gather(
                        *[
                            block.aput(
                                messages_to_flush,
                                from_short_term_memory=True,
                                session_id=self.session_id,
                            )
                            for block in self.memory_blocks
                        ]
                    )

                # Recalculate remaining tokens
                chronological_view = reversed_queue[::-1]
                tokens_in_current_queue = sum(
                    self._estimate_token_count(message)
                    for message in chronological_view
                )
                tokens_to_remove = tokens_in_current_queue - token_limit

                # Exit if we've flushed everything possible but still over limit
                if not messages_to_flush:
                    break

    async def aput(self, message: ChatMessage) -> None:
        """Add a message to the chat store and process waterfall logic if needed."""
        # Add the message to the chat store
        await self.sql_store.add_message(
            self.session_id, message, status=MessageStatus.ACTIVE
        )

        # Ensure the active queue is managed
        await self._manage_queue()

    async def aput_messages(self, messages: List[ChatMessage]) -> None:
        """Add a list of messages to the chat store and process waterfall logic if needed."""
        # Add the messages to the chat store
        await self.sql_store.add_messages(
            self.session_id, messages, status=MessageStatus.ACTIVE
        )

        # Ensure the active queue is managed
        await self._manage_queue()

    async def aset(self, messages: List[ChatMessage]) -> None:
        """Set the chat history."""
        await self.sql_store.set_messages(
            self.session_id, messages, status=MessageStatus.ACTIVE
        )

    async def aget_all(
        self, status: Optional[MessageStatus] = None
    ) -> List[ChatMessage]:
        """Get all messages."""
        return await self.sql_store.get_messages(self.session_id, status=status)

    async def areset(self, status: Optional[MessageStatus] = None) -> None:
        """Reset the memory."""
        await self.sql_store.delete_messages(self.session_id, status=status)

    # ---- Sync method wrappers ----

    def get(
        self, input: Optional[Union[str, ChatMessage]] = None, **block_kwargs: Any
    ) -> List[ChatMessage]:  # type: ignore[override]
        """Get messages with memory blocks included."""
        return asyncio_run(self.aget(input=input, **block_kwargs))

    def get_all(self, status: Optional[MessageStatus] = None) -> List[ChatMessage]:
        """Get all messages."""
        return asyncio_run(self.aget_all(status=status))

    def put(self, message: ChatMessage) -> None:
        """Add a message to the chat store and process waterfall logic if needed."""
        return asyncio_run(self.aput(message))

    def set(self, messages: List[ChatMessage]) -> None:
        """Set the chat history."""
        return asyncio_run(self.aset(messages))

    def reset(self) -> None:
        """Reset the memory."""
        return asyncio_run(self.areset())
