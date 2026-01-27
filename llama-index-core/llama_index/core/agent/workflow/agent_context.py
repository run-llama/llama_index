"""
Agent context protocol and simple implementation for non-workflow usage.

This module provides a minimal duck-typed protocol that `take_step` implementations
use, allowing agents to work both in full workflow contexts and in simpler
scenarios like `LLM.predict_and_call`.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from workflows.context.state_store import DictState, InMemoryStateStore


@runtime_checkable
class AgentContext(Protocol):
    """
    Minimal context interface for agent take_step implementations.

    This protocol defines the subset of Context that agents actually use,
    allowing for both full workflow Context and lightweight alternatives.
    """

    @property
    def store(self) -> InMemoryStateStore[Any]:
        """Access the key-value store for agent state."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if the workflow is actively running (for event writing)."""
        ...

    def write_event_to_stream(self, event: Any) -> None:
        """Write an event to the output stream."""
        ...


def _default_store() -> InMemoryStateStore[DictState]:
    return InMemoryStateStore(DictState())


@dataclass(frozen=True)
class SimpleAgentContext:
    """
    Lightweight context for agents used outside workflows.

    This implementation satisfies the AgentContext protocol with minimal
    overhead, suitable for use in `LLM.predict_and_call` and similar
    non-workflow scenarios where a full Context is not needed.
    """

    store: InMemoryStateStore[DictState] = field(default_factory=_default_store)
    is_running: bool = False

    def write_event_to_stream(self, event: Any) -> None:
        """No-op - events are discarded in non-workflow usage."""
