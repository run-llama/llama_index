"""
Agent context protocol and simple implementation for non-workflow usage.

This module provides a minimal duck-typed protocol that `take_step` implementations
use, allowing agents to work both in full workflow contexts and in simpler
scenarios like `LLM.predict_and_call`.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentContextStore(Protocol):
    """Protocol for agent state storage."""

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        ...


@runtime_checkable
class AgentContext(Protocol):
    """
    Minimal context interface for agent take_step implementations.

    This protocol defines the subset of Context that agents actually use,
    allowing for both full workflow Context and lightweight alternatives.
    """

    @property
    def store(self) -> AgentContextStore:
        """Access the key-value store for agent state."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if the workflow is actively running (for event writing)."""
        ...

    def write_event_to_stream(self, event: Any) -> None:
        """Write an event to the output stream."""
        ...


class SimpleAgentContextStore:
    """Simple dict-based store for non-workflow agent usage."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        self._data[key] = value


class SimpleAgentContext:
    """
    Lightweight context for agents used outside workflows.

    This implementation satisfies the AgentContext protocol with minimal
    overhead, suitable for use in `LLM.predict_and_call` and similar
    non-workflow scenarios where a full Context is not needed.
    """

    def __init__(self) -> None:
        self._store = SimpleAgentContextStore()

    @property
    def store(self) -> SimpleAgentContextStore:
        """Access the key-value store for agent state."""
        return self._store

    @property
    def is_running(self) -> bool:
        """
        Always returns False - events should be skipped.

        In non-workflow usage, there's no event stream to write to,
        so agents should skip event writing when this returns False.
        """
        return False

    def write_event_to_stream(self, event: Any) -> None:
        """No-op - events are discarded in non-workflow usage."""
