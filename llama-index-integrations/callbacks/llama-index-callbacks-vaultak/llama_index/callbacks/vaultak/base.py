"""
Vaultak runtime security callback handler for LlamaIndex.

Intercepts every agent action, LLM call, tool use, and query in real time,
routing them through Vaultak's risk scoring and policy engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

logger = logging.getLogger(__name__)


class VaultakCallbackHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler that wraps every agent action with
    Vaultak runtime security monitoring.

    Usage:
        from llama_index.core.callbacks import CallbackManager
        from llama_index.callbacks.vaultak import VaultakCallbackHandler

        handler = VaultakCallbackHandler(api_key="vtk_...")
        callback_manager = CallbackManager([handler])

        # Use with a query engine
        query_engine = index.as_query_engine(
            callback_manager=callback_manager
        )

        # Or set globally
        from llama_index.core import Settings
        Settings.callback_manager = callback_manager
    """

    def __init__(
        self,
        api_key: str,
        agent_name: str = "llamaindex-agent",
        block_on_high_risk: bool = True,
        risk_threshold: float = 7.0,
        verbose: bool = False,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ):
        """
        Initialize the Vaultak callback handler.

        Args:
            api_key: Your Vaultak API key (starts with vtk_)
            agent_name: Human-readable name for this agent in the Vaultak dashboard
            block_on_high_risk: If True, raises an exception when risk score
                                 exceeds risk_threshold
            risk_threshold: Score (0-10) above which actions are blocked (default 7.0)
            verbose: Log all scored actions to stdout
            event_starts_to_ignore: List of CBEventTypes to ignore on start
            event_ends_to_ignore: List of CBEventTypes to ignore on end
        """
        try:
            from vaultak import Vaultak
        except ImportError:
            raise ImportError(
                "vaultak is required. Install it with: pip install vaultak"
            )

        self.vt = Vaultak(api_key=api_key)
        self.agent_name = agent_name
        self.block_on_high_risk = block_on_high_risk
        self.risk_threshold = risk_threshold
        self.verbose = verbose
        self._active_trace: Optional[str] = None

        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a Vaultak monitoring session when a trace begins."""
        self._active_trace = trace_id
        try:
            self._monitor_ctx = self.vt.monitor(
                self.agent_name if not trace_id else f"{self.agent_name}:{trace_id}"
            )
            self._monitor_ctx.__enter__()
        except Exception as e:
            logger.warning(f"[Vaultak] Could not start monitor: {e}")

        if self.verbose:
            logger.info(f"[Vaultak] Trace started: {trace_id}")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End the Vaultak monitoring session when a trace ends."""
        try:
            if hasattr(self, "_monitor_ctx") and self._monitor_ctx:
                self._monitor_ctx.__exit__(None, None, None)
                self._monitor_ctx = None
        except Exception as e:
            logger.warning(f"[Vaultak] Could not end monitor: {e}")

        if self.verbose:
            logger.info(f"[Vaultak] Trace ended: {trace_id}")

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Intercept events at start — score and enforce policy on agent actions."""
        payload = payload or {}

        # Score function/tool calls before they execute
        if event_type == CBEventType.FUNCTION_CALL:
            function_call = payload.get(EventPayload.FUNCTION_CALL, "unknown")
            tool_name = str(function_call)

            if self.verbose:
                logger.info(f"[Vaultak] Scoring function call: {tool_name}")

            try:
                result = self.vt.score_action(
                    action=tool_name,
                    context={"payload": str(payload), "agent": self.agent_name},
                )
                risk_score = getattr(result, "score", 0)

                if self.verbose:
                    logger.info(f"[Vaultak] Risk score for '{tool_name}': {risk_score}/10")

                if self.block_on_high_risk and risk_score >= self.risk_threshold:
                    raise RuntimeError(
                        f"[Vaultak] Function call '{tool_name}' blocked — "
                        f"risk score {risk_score:.1f} exceeds threshold {self.risk_threshold}. "
                        f"Review in your Vaultak dashboard at app.vaultak.com"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"[Vaultak] Could not score function call '{tool_name}': {e}")

        # Check LLM inputs for policy compliance
        elif event_type == CBEventType.LLM:
            messages = payload.get(EventPayload.MESSAGES, [])
            if messages:
                try:
                    self.vt.check_policy(
                        tool_name="llm_call",
                        input_data=str(messages),
                    )
                except Exception as e:
                    if "blocked" in str(e).lower() or "denied" in str(e).lower():
                        raise RuntimeError(f"[Vaultak] LLM call blocked by policy: {e}")

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Scan event outputs — mask PII and handle exceptions."""
        payload = payload or {}

        # Mask PII in function call outputs
        if event_type == CBEventType.FUNCTION_CALL:
            response = payload.get(EventPayload.FUNCTION_OUTPUT)
            if response:
                try:
                    masked = self.vt.mask_pii(str(response))
                    if self.verbose and masked != str(response):
                        logger.info(f"[Vaultak] PII masked in function output")
                except Exception:
                    pass

        # Alert on exceptions
        elif event_type == CBEventType.EXCEPTION:
            exception = payload.get(EventPayload.EXCEPTION)
            if exception:
                try:
                    self.vt.alert(
                        level="high",
                        message=f"LlamaIndex exception: {type(exception).__name__}: {str(exception)}",
                    )
                    self.vt.rollback(reason=str(exception))
                except Exception:
                    pass

        # Mask PII in query responses
        elif event_type == CBEventType.QUERY:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                try:
                    self.vt.mask_pii(str(response))
                except Exception:
                    pass
