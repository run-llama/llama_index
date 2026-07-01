"""Dakera persistent memory integration for LlamaIndex.

Dakera provides decay-weighted, cross-session memory with semantic recall
via a REST API. This integration stores and retrieves chat messages using
the Dakera memory server, giving agents persistent, relevance-ranked memory
without any additional dependencies beyond httpx.
"""

from typing import Any, List, Optional

import httpx
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.memory import BaseMemory


class DakeraMemory(BaseMemory):
    """Dakera persistent memory for LlamaIndex agents.

    Stores and retrieves chat messages via the Dakera REST API, which
    provides decay-weighted vector memory with semantic search. Memories
    persist across sessions and are automatically ranked by recency and
    relevance.

    Args:
        base_url: Base URL of the Dakera API server (e.g. ``"https://api.dakera.ai"``).
        api_key: API key for authenticating with the Dakera server.
        session_id: Unique identifier for the current session. Memories are
            namespaced by session so different agent runs stay isolated.
        top_k: Maximum number of memories to retrieve per search (default 10).

    Example:
        .. code-block:: python

            from llama_index.memory.dakera import DakeraMemory
            from llama_index.core import SimpleChatEngine
            from llama_index.llms.openai import OpenAI

            memory = DakeraMemory(
                base_url="https://api.dakera.ai",
                api_key="dak-...",
                session_id="user-123",
                top_k=10,
            )

            llm = OpenAI(model="gpt-4o-mini")
            engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
            response = engine.chat("What did we discuss last time?")
    """

    base_url: str = Field(description="Base URL of the Dakera API server.")
    api_key: str = Field(description="API key for the Dakera server.")
    session_id: str = Field(description="Session identifier for memory namespacing.")
    top_k: int = Field(
        default=10,
        description="Maximum number of memories to retrieve per semantic search.",
    )

    _client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        api_key: str,
        session_id: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            session_id=session_id,
            top_k=top_k,
            **kwargs,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for serialization."""
        return "DakeraMemory"

    @classmethod
    def from_defaults(
        cls,
        base_url: str = "https://api.dakera.ai",
        api_key: str = "",
        session_id: str = "default",
        top_k: int = 10,
        **kwargs: Any,
    ) -> "DakeraMemory":
        """Create a DakeraMemory instance with sensible defaults.

        Args:
            base_url: Base URL of the Dakera API server.
            api_key: API key for the Dakera server.
            session_id: Session identifier for memory namespacing.
            top_k: Maximum number of memories to retrieve per search.

        Returns:
            DakeraMemory instance.
        """
        return cls(
            base_url=base_url,
            api_key=api_key,
            session_id=session_id,
            top_k=top_k,
            **kwargs,
        )

    async def get(
        self, input: Optional[str] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Retrieve relevant memories for the given input via semantic search.

        Performs a semantic similarity search against stored memories and
        returns the top-k most relevant results as ChatMessages. If ``input``
        is ``None``, returns an empty list.

        Args:
            input: The query string to search memories against (typically
                the current user message or conversation context).

        Returns:
            List of ChatMessage objects representing relevant past memories,
            ordered by relevance score descending.
        """
        if not input:
            return []

        try:
            response = await self._client.post(
                "/v1/memories/search",
                json={
                    "query": input,
                    "session_id": self.session_id,
                    "top_k": self.top_k,
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        results = data if isinstance(data, list) else data.get("results", [])
        messages: List[ChatMessage] = []
        for item in results:
            content = item.get("content") if isinstance(item, dict) else str(item)
            if content:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=str(content))
                )
        return messages

    async def get_all(self) -> List[ChatMessage]:
        """Retrieve all stored memories for the current session.

        Performs a broad semantic search using an empty query to surface all
        available memories for the session, up to ``top_k`` results.

        Returns:
            List of ChatMessage objects for all stored memories in the session.
        """
        try:
            response = await self._client.post(
                "/v1/memories/search",
                json={
                    "query": "",
                    "session_id": self.session_id,
                    "top_k": self.top_k,
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        results = data if isinstance(data, list) else data.get("results", [])
        messages: List[ChatMessage] = []
        for item in results:
            content = item.get("content") if isinstance(item, dict) else str(item)
            if content:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=str(content))
                )
        return messages

    async def put(self, message: ChatMessage) -> None:
        """Store a single ChatMessage in Dakera memory.

        The message content is persisted with the current session identifier.
        Dakera automatically applies decay weighting so recent, frequently
        accessed memories surface higher in future recalls.

        Args:
            message: The ChatMessage to store.
        """
        content = message.content
        if not content:
            return

        try:
            response = await self._client.post(
                "/v1/memories",
                json={
                    "content": str(content),
                    "session_id": self.session_id,
                },
            )
            response.raise_for_status()
        except Exception:
            pass

    async def set(self, messages: List[ChatMessage]) -> None:
        """Replace all session memories with the given message list.

        Clears existing memories for the session and stores all provided
        messages. Useful for initializing memory from a known conversation
        history.

        Args:
            messages: List of ChatMessages to store as the full memory state.
        """
        await self.reset()
        for message in messages:
            await self.put(message)

    async def reset(self) -> None:
        """Delete all memories for the current session.

        Permanently removes all stored memories associated with the current
        ``session_id``. This cannot be undone.
        """
        try:
            response = await self._client.delete(
                "/v1/memories",
                json={"session_id": self.session_id},
            )
            response.raise_for_status()
        except Exception:
            pass
