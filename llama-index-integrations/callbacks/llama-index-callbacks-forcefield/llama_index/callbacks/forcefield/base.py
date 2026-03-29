"""ForceField AI security callback handler for LlamaIndex.

Scans prompts before they reach the LLM and moderates outputs after generation.

Usage::

    from llama_index.core import Settings
    from llama_index.callbacks.forcefield import ForceFieldCallbackHandler

    handler = ForceFieldCallbackHandler(sensitivity="high")
    Settings.callback_manager.add_handler(handler)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from forcefield import Guard

logger = logging.getLogger(__name__)


class PromptBlockedError(Exception):
    """Raised when ForceField blocks a prompt in the LlamaIndex pipeline."""

    def __init__(self, message: str, scan_result: Any = None) -> None:
        super().__init__(message)
        self.scan_result = scan_result


class ForceFieldCallbackHandler(BaseCallbackHandler):
    """LlamaIndex callback that scans prompts and moderates outputs.

    Args:
        sensitivity: Detection sensitivity level (low, medium, high, critical).
        block_on_input: Raise PromptBlockedError if input is blocked.
        moderate_output: Run output moderation on LLM responses.
        on_block: Optional callable for custom block handling.
    """

    def __init__(
        self,
        sensitivity: str = "medium",
        block_on_input: bool = True,
        moderate_output: bool = True,
        on_block: Optional[Any] = None,
        **guard_kwargs: Any,
    ) -> None:
        super().__init__(
            event_starts_to_trace=[CBEventType.LLM],
            event_ends_to_trace=[CBEventType.LLM],
        )
        self.guard = Guard(sensitivity=sensitivity, **guard_kwargs)
        self.block_on_input = block_on_input
        self.moderate_output = moderate_output
        self.on_block = on_block
        self._last_result: Any = None

    @property
    def last_result(self) -> Any:
        """The most recent scan result."""
        return self._last_result

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace (no-op)."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace (no-op)."""

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Scan prompts before they reach the LLM."""
        if event_type == CBEventType.LLM and payload:
            prompt = None
            if EventPayload.PROMPT in payload:
                prompt = payload[EventPayload.PROMPT]
            elif EventPayload.MESSAGES in payload:
                messages = payload[EventPayload.MESSAGES]
                if messages:
                    last = messages[-1]
                    prompt = getattr(last, "content", None) or str(last)

            if prompt:
                result = self.guard.scan(prompt)
                self._last_result = result
                if result.blocked:
                    logger.warning(
                        "ForceField blocked prompt: risk=%.2f rules=%s",
                        result.risk_score,
                        result.rules_triggered,
                    )
                    if self.on_block:
                        self.on_block(result)
                    if self.block_on_input:
                        raise PromptBlockedError(
                            f"ForceField blocked: {', '.join(result.rules_triggered)}",
                            scan_result=result,
                        )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Moderate LLM outputs for harmful content."""
        if not self.moderate_output:
            return
        if event_type == CBEventType.LLM and payload:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                text = str(response)
                if text:
                    try:
                        mod = self.guard.moderate(text)
                        if not mod.passed:
                            logger.warning(
                                "ForceField moderation flagged output: action=%s categories=%s",
                                mod.action.value,
                                mod.categories,
                            )
                    except Exception:
                        pass
