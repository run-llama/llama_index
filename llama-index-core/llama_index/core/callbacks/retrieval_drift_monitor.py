"""Retrieval Drift Monitor callback for LlamaIndex.

Monitors embedding distribution shift in retrieval pipelines using the
Bhattacharyya distance between rolling windows of query embeddings.  Fires
a configurable alert when drift exceeds a threshold.

Ported from kernel-ml/driftwatch (Sathyavageeswaran, 2024-2026).
Reference: US Patent 19/287,703; IJITCE Vol. 13 (2025).
"""

from __future__ import annotations

import collections
import math
from typing import Any, Callable, Deque, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload


class RetrievalDriftMonitor(BaseCallbackHandler):
    """Monitors query embedding distribution shift via Bhattacharyya distance.

    Maintains two rolling windows of query embeddings (reference and current).
    When the current window fills, it computes the Bhattacharyya distance between
    the mean embedding distributions.  If distance exceeds ``bc_alert_threshold``,
    ``alert_fn`` is called with a descriptive message.

    The Bhattacharyya distance (BD) between two distributions P and Q is:
    ``BD(P, Q) = -ln(BC(P, Q))``
    where ``BC = Σ sqrt(p_i * q_i)`` (Bhattacharyya Coefficient).

    BD = 0 means identical distributions; larger values indicate more divergence.

    Args:
        window_size: Number of query embeddings per window (default 100).
        bc_alert_threshold: Bhattacharyya distance threshold for firing an alert
            (default 0.15).  Lower = more sensitive.
        alert_fn: Callable that receives the alert message string.  Defaults to
            ``print``.

    Example::

        from llama_index.core.callbacks import CallbackManager
        from llama_index.core.callbacks import RetrievalDriftMonitor

        def my_alert(msg: str):
            logging.warning(msg)

        monitor = RetrievalDriftMonitor(window_size=50, bc_alert_threshold=0.1,
                                        alert_fn=my_alert)
        callback_manager = CallbackManager([monitor])

    """

    def __init__(
        self,
        window_size: int = 100,
        bc_alert_threshold: float = 0.15,
        alert_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.window_size = window_size
        self.bc_alert_threshold = bc_alert_threshold
        self.alert_fn: Callable[[str], None] = alert_fn or (
            lambda msg: print(f"[RetrievalDriftMonitor] {msg}")
        )
        self._ref_window: Deque[List[float]] = collections.deque(maxlen=window_size)
        self._cur_window: Deque[List[float]] = collections.deque(maxlen=window_size)
        self._alert_count: int = 0

    # ------------------------------------------------------------------
    # BaseCallbackHandler interface
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != CBEventType.RETRIEVE or payload is None:
            return

        # Extract query embedding from payload if available.
        # The RETRIEVE end event may carry the query embedding under
        # EventPayload.EMBEDDINGS or a custom key; we try both.
        embedding: Optional[List[float]] = None
        if EventPayload.EMBEDDINGS in payload:
            raw = payload[EventPayload.EMBEDDINGS]
            if isinstance(raw, list) and raw:
                embedding = raw[0] if isinstance(raw[0], list) else raw
        if embedding is None:
            embedding = payload.get("query_embedding")

        if embedding is None or len(embedding) == 0:
            return

        self._cur_window.append(list(embedding))
        if len(self._cur_window) == self.window_size:
            self._check_drift()

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Total number of drift alerts fired since instantiation."""
        return self._alert_count

    def _mean_embedding(self, window: Deque[List[float]]) -> List[float]:
        if not window:
            return []
        n = len(window[0])
        mean = [0.0] * n
        for emb in window:
            for i, v in enumerate(emb):
                mean[i] += v
        total = len(window)
        return [v / total for v in mean]

    @staticmethod
    def _bhattacharyya_distance(p: List[float], q: List[float]) -> float:
        """Compute Bhattacharyya distance between two unnormalised vectors."""
        if len(p) != len(q) or not p:
            return 0.0
        sum_p = sum(abs(v) for v in p) + 1e-10
        sum_q = sum(abs(v) for v in q) + 1e-10
        bc = sum(
            math.sqrt((abs(p[i]) / sum_p) * (abs(q[i]) / sum_q))
            for i in range(len(p))
        )
        bc = max(bc, 1e-10)
        return -math.log(bc)

    def _check_drift(self) -> None:
        cur_mean = self._mean_embedding(self._cur_window)
        if not self._ref_window:
            # First window becomes the reference baseline
            self._ref_window.extend(self._cur_window)
            self._cur_window.clear()
            return

        ref_mean = self._mean_embedding(self._ref_window)
        bd = self._bhattacharyya_distance(ref_mean, cur_mean)

        if bd > self.bc_alert_threshold:
            self._alert_count += 1
            self.alert_fn(
                f"Retrieval drift detected: Bhattacharyya distance {bd:.4f} "
                f"exceeds threshold {self.bc_alert_threshold:.4f} "
                f"(alert #{self._alert_count})"
            )

        # Rotate: current window becomes the new reference
        self._ref_window.clear()
        self._ref_window.extend(self._cur_window)
        self._cur_window.clear()

    def reset(self) -> None:
        """Clear both windows and reset the alert counter."""
        self._ref_window.clear()
        self._cur_window.clear()
        self._alert_count = 0
