"""Primordia callback for LlamaIndex cost tracking."""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType


class PrimordiaCallbackHandler(BaseCallbackHandler):
    """Emit MSR receipts for LlamaIndex operations.

    Shadow mode by default - no network calls, no blocking.

    Example:
        >>> from llama_index.core.callbacks import PrimordiaCallbackHandler
        >>> handler = PrimordiaCallbackHandler(agent_id="rag-agent")
        >>> Settings.callback_manager.add_handler(handler)
    """

    def __init__(self, agent_id: str, kernel_url: str = "https://clearing.kaledge.app"):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.agent_id = agent_id
        self.kernel_url = kernel_url
        self.receipts: List[Dict] = []

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Emit MSR receipt on LLM event completion."""
        if event_type == CBEventType.LLM and payload:
            tokens = payload.get("total_tokens", 0)
            model = payload.get("model", "unknown")

            unit_price = 80
            if "gpt-4" in str(model).lower():
                unit_price = 300
            elif "claude" in str(model).lower():
                unit_price = 100

            receipt = {
                "meter_version": "0.1",
                "type": "compute",
                "agent_id": self.agent_id,
                "provider": str(model),
                "units": tokens,
                "unit_price_usd_micros": unit_price,
                "total_usd_micros": tokens * unit_price,
                "timestamp_ms": int(time.time() * 1000),
                "metadata": {"framework": "llamaindex", "event_id": event_id}
            }

            receipt_hash = hashlib.sha256(
                json.dumps(receipt, sort_keys=True).encode()
            ).hexdigest()[:32]

            self.receipts.append({"hash": receipt_hash, "receipt": receipt})

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """No-op on event start."""
        return event_id

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        return sum(r["receipt"]["total_usd_micros"] for r in self.receipts) / 1_000_000

    def get_receipts(self) -> List[Dict]:
        """Get all receipts for settlement."""
        return self.receipts

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """No-op trace start."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """No-op trace end."""
        pass
