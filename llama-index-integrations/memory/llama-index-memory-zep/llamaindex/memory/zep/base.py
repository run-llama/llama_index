from typing import List, Optional, Dict, Any, Union
import uuid
from llama_index.core.memory.types import BaseMemory
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from zep_python.types import Message


class ZepMemory(BaseMemory):
    """Zep Memory for LlamaIndex."""
    
    session_id: str = Field(description="Zep session ID")
    user_id: Optional[str] = Field(default=None, description="User ID for user-specific context")
    memory_key: str = Field(default="chat_history", description="Memory key for prompt")
    max_message_length: int = Field(default=2500, description="Maximum character length for messages")
    is_async: bool = Field(default=False, description="Whether to use AsyncZep client")
    
    # Private attributes
    _client = PrivateAttr(default=None)
    _primary_memory: BaseMemory = PrivateAttr(default=None)
    
    def __init__(
        self,
        session_id: str,
        zep_client,
        user_id: Optional[str] = None,
        memory_key: str = "chat_history",
        max_message_length: int = 2500,
        is_async: bool = False 
    ):
        """Initialize with Zep client and session."""
        super().__init__(
            session_id=session_id,
            user_id=user_id,
            memory_key=memory_key,
            max_message_length=max_message_length,
            is_async=is_async
        )
        self._client = zep_client
        self._primary_memory = ChatMemoryBuffer.from_defaults()
        self._sync_from_zep()
    
    @classmethod
    def class_name(cls) -> str:
        return "ZepMemory"
    
    @classmethod
    def from_defaults(
        cls, 
        zep_client=None, 
        session_id=None, 
        user_id: Optional[str] = None, 
        is_async: bool =False
    ):
        if zep_client is None:
            raise ValueError("zep_client is required")
        if session_id is None:
            session_id = str(uuid.uuid4())
        return cls(
            zep_client=zep_client,
            session_id=session_id,
            user_id=user_id,  # Now you can pass the user ID
            memory_key="chat_history",
            max_message_length=2500,
            is_async=is_async
        )

    
    def _convert_to_zep_message(self, message: ChatMessage) -> Any:
        role_map = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.SYSTEM: "system",
            MessageRole.TOOL: "tool",
        }
        role = role_map.get(message.role, "user")
        content = message.content if message.content is not None else ""
        if len(content) > self.max_message_length:
            content = content[:self.max_message_length]
        return Message(
            role=role,
            content=content,
            role_type=role,
            metadata=message.additional_kwargs or {},
        )
    
    def _sync_from_zep(self) -> None:
        """Synchronously retrieve memory from Zep and update local memory."""
        if self._client is None:
            return
        try:
            zep_memory = self._client.memory.get(session_id=self.session_id)
            if not zep_memory:
                return
            messages: List[ChatMessage] = []
            if hasattr(zep_memory, "messages") and zep_memory.messages:
                for msg in zep_memory.messages:
                    role_map = {
                        "user": MessageRole.USER,
                        "assistant": MessageRole.ASSISTANT,
                        "system": MessageRole.SYSTEM,
                        "tool": MessageRole.TOOL,
                        "function": MessageRole.TOOL,
                    }
                    role_str = getattr(msg, "role", None)
                    if not role_str and hasattr(msg, "role_type"):
                        role_str = msg.role_type
                    role_str = role_str.lower() if role_str else "user"
                    role = role_map.get(role_str, MessageRole.USER)
                    content = getattr(msg, "content", "")
                    metadata = getattr(msg, "metadata", {}) or {}
                    chat_message = ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=metadata,
                    )
                    messages.append(chat_message)
            if messages:
                self._primary_memory.set(messages)
        except Exception:
            pass  # Silently ignore errors during sync
    
    async def _async_sync_from_zep(self) -> None:
        """Asynchronously retrieve memory from Zep and update local memory."""
        if self._client is None or not self.is_async:
            return
        try:
            zep_memory = await self._client.memory.get(session_id=self.session_id)
            if not zep_memory:
                return
            messages: List[ChatMessage] = []
            if hasattr(zep_memory, "messages") and zep_memory.messages:
                for msg in zep_memory.messages:
                    role_map = {
                        "user": MessageRole.USER,
                        "assistant": MessageRole.ASSISTANT,
                        "system": MessageRole.SYSTEM,
                        "tool": MessageRole.TOOL,
                        "function": MessageRole.TOOL,
                    }
                    role_str = getattr(msg, "role", None)
                    if not role_str and hasattr(msg, "role_type"):
                        role_str = msg.role_type
                    role_str = role_str.lower() if role_str else "user"
                    role = role_map.get(role_str, MessageRole.USER)
                    content = getattr(msg, "content", "")
                    metadata = getattr(msg, "metadata", {}) or {}
                    chat_message = ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=metadata,
                    )
                    messages.append(chat_message)
            if messages:
                self._primary_memory.set(messages)
        except Exception:
            pass  # Silently ignore errors during sync
    
    def _get_context_from_memory(self, query: Optional[str] = None) -> str:
        """Retrieve and compile context from Zep memory."""
        if self._client is None:
            return ""
        try:
            zep_memory = self._client.memory.get(session_id=self.session_id)
            context_parts: List[str] = []
            if hasattr(zep_memory, "facts") and zep_memory.facts:
                context_parts.append("Facts:")
                for fact in zep_memory.facts:
                    context_parts.append(f"- {fact}")
            if (hasattr(zep_memory, "summary") and zep_memory.summary and
                hasattr(zep_memory.summary, "content") and zep_memory.summary.content):
                context_parts.append("\nSummary:")
                context_parts.append(zep_memory.summary.content)
            if hasattr(zep_memory, "context") and zep_memory.context:
                context_parts.append("\nContext:")
                context_parts.append(zep_memory.context)
            if query and self.user_id:
                try:
                    edge_results = self._client.memory.search_sessions(
                        user_id=self.user_id,
                        text=query,
                        search_scope="edges",
                        limit=5,
                    )
                    if edge_results and hasattr(edge_results, "edges") and edge_results.edges:
                        context_parts.append("\nRelevant information:")
                        for edge in edge_results.edges:
                            if hasattr(edge, "fact"):
                                context_parts.append(f"- {edge.fact}")
                except Exception:
                    pass
            return "\n".join(context_parts)
        except Exception:
            return ""
        
    async def _async_get_context_from_memory(self, query: Optional[str] = None) -> str:
        """Asynchronously retrieve and compile context from Zep memory."""
        if self._client is None or not self.is_async:
            return ""
        try:
            zep_memory = await self._client.memory.get(session_id=self.session_id)
            context_parts: List[str] = []
            if hasattr(zep_memory, "facts") and zep_memory.facts:
                context_parts.append("Facts:")
                for fact in zep_memory.facts:
                    context_parts.append(f"- {fact}")
            if (hasattr(zep_memory, "summary") and zep_memory.summary and
                hasattr(zep_memory.summary, "content") and zep_memory.summary.content):
                context_parts.append("\nSummary:")
                context_parts.append(zep_memory.summary.content)
            if hasattr(zep_memory, "context") and zep_memory.context:
                context_parts.append("\nContext:")
                context_parts.append(zep_memory.context)
            if query and self.user_id:
                try:
                    edge_results = await self._client.memory.search_sessions(
                        user_id=self.user_id,
                        text=query,
                        search_scope="edges",
                        limit=5,
                    )
                    if edge_results and hasattr(edge_results, "edges") and edge_results.edges:
                        context_parts.append("\nRelevant information:")
                        for edge in edge_results.edges:
                            if hasattr(edge, "fact"):
                                context_parts.append(f"- {edge.fact}")
                except Exception:
                    pass
            return "\n".join(context_parts)
        except Exception:
            return ""
    
    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """Retrieve chat history with context enrichment."""
        messages = self._primary_memory.get(input=input, **kwargs)
        if self._client is None:
            return messages
        context = self._get_context_from_memory(input)
        if context:
            if messages and messages[0].role == MessageRole.SYSTEM:
                updated_content = f"{messages[0].content}\n\n{context}"
                messages[0] = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=updated_content,
                    additional_kwargs=messages[0].additional_kwargs,
                )
            else:
                system_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=context,
                )
                messages.insert(0, system_message)
        return messages
    
    async def aget(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """Asynchronously retrieve chat history with context enrichment."""
        if not self.is_async:
            raise ValueError("Cannot use asynchronous aget() when is_async=False. Use get() instead.")
            
        # First sync from Zep to ensure we have the latest data
        await self._async_sync_from_zep()
        
        # Now get the messages from the primary memory
        if hasattr(self._primary_memory, "aget"):
            messages = await self._primary_memory.aget(input=input, **kwargs)
        else:
            messages = self._primary_memory.get(input=input, **kwargs)
            
        if self._client is None:
            return messages
            
        context = await self._async_get_context_from_memory(input)
        if context:
            if messages and messages[0].role == MessageRole.SYSTEM:
                updated_content = f"{messages[0].content}\n\n{context}"
                messages[0] = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=updated_content,
                    additional_kwargs=messages[0].additional_kwargs,
                )
            else:
                system_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=context,
                )
                messages.insert(0, system_message)
        return messages
    
    def get_all(self) -> List[ChatMessage]:
        """Retrieve all chat history without context enrichment."""
        return self._primary_memory.get_all()
    
    async def aget_all(self) -> List[ChatMessage]:
        """Asynchronously retrieve all chat history without context enrichment."""
        # First sync from Zep
        await self._async_sync_from_zep()
        
        if hasattr(self._primary_memory, "aget_all"):
            return await self._primary_memory.aget_all()
        return self._primary_memory.get_all()
    
    def _add_msgs_to_zep(self, messages: List[ChatMessage]) -> None:
        """Add new messages to Zep  memory with truncation."""
        if self._client is None or not messages:
            return
        try:
            zep_messages = []
            for msg in messages:
                zep_msg = self._convert_to_zep_message(msg)
                if hasattr(zep_msg, "content") and zep_msg.content:
                    if len(zep_msg.content) > self.max_message_length:
                        zep_msg.content = zep_msg.content[:self.max_message_length - 3] + "..."
                zep_messages.append(zep_msg)
            try:
                self._client.memory.get(session_id=self.session_id)
            except Exception:
                pass
            if zep_messages:
                self._client.memory.add(
                    session_id=self.session_id,
                    messages=zep_messages,
                )
        except Exception:
            pass

    async def _async_add_msgs_to_zep(self, messages: List[ChatMessage]) -> None:
            """Asynchronously add new messages to Zep memory with truncation."""
            if self._client is None or not messages or not self.is_async:
                return
            try:
                zep_messages = []
                for msg in messages:
                    zep_msg = self._convert_to_zep_message(msg)
                    if hasattr(zep_msg, "content") and zep_msg.content:
                        if len(zep_msg.content) > self.max_message_length:
                            zep_msg.content = zep_msg.content[:self.max_message_length - 3] + "..."
                    zep_messages.append(zep_msg)
                try:
                    await self._client.memory.get(session_id=self.session_id)
                except Exception:
                    pass
                if zep_messages:
                    await self._client.memory.add(
                        session_id=self.session_id,
                        messages=zep_messages,
                    )
            except Exception:
                pass 
    def put(self, message: ChatMessage) -> None:
        """Add a message to memory."""
        self._primary_memory.put(message)
        self._add_msgs_to_zep([message])
    
    async def aput(self, message: ChatMessage) -> None:
        """Asynchronously add a message to memory."""
        if hasattr(self._primary_memory, "aput"):
            await self._primary_memory.aput(message)
        else:
            self._primary_memory.put(message)
        await self._async_add_msgs_to_zep([message])
    
    def set(self, messages: List[ChatMessage]) -> None:
        """Replace the entire chat history."""
        initial_chat_len = len(self._primary_memory.get_all())
        self._primary_memory.set(messages)
        if len(messages) > initial_chat_len:
            self._add_msgs_to_zep(messages[initial_chat_len:])
    
    async def aset(self, messages: List[ChatMessage]) -> None:
        """Asynchronously replace the entire chat history."""
        initial_chat_len = len(self._primary_memory.get_all())
        self._primary_memory.set(messages)  # No async version typically available
        if len(messages) > initial_chat_len:
            await self._async_add_msgs_to_zep(messages[initial_chat_len:])

    def reset(self) -> None:
        """Clear the memory."""
        self._primary_memory.reset()
        if self._client is not None:
            try:
                self._client.memory.delete(session_id=self.session_id)
            except Exception:
                pass
    
    async def areset(self) -> None:
        """Asynchronously clear the memory."""
        if hasattr(self._primary_memory, "areset"):
            await self._primary_memory.areset()
        else:
            self._primary_memory.reset()
        if self._client is not None:
            try:
                await self._client.memory.delete(session_id=self.session_id)
            except Exception:
                pass

    def search(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search memory for relevant content."""
        if self._client is None:
            raise ValueError("Client is not initialized")
        try:
            return self._client.memory.search_sessions(
                session_ids=[self.session_id],
                user_id=self.user_id,
                text=query,
                **kwargs
            )
        except Exception:
            return None

    async def asearch(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
            """Asynchronously search memory for relevant content."""
            if self._client is None:
                raise ValueError("Client is not initialized")
            try:
                return await self._client.memory.search_sessions(
                    session_ids=[self.session_id],
                    user_id=self.user_id,
                    text=query,
                    **kwargs
                )
            except Exception:
                return None