from typing import Any, Dict, List, Optional, cast

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload


class SimpleLLMHandler(BaseCallbackHandler):
    """Callback handler for printing llms inputs/outputs."""

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    def _print_llm_event(self, payload: dict) -> None:
        from llama_index.core.llms import ChatMessage

        if EventPayload.PROMPT in payload:
            prompt = str(payload.get(EventPayload.PROMPT))
            completion = str(payload.get(EventPayload.COMPLETION))

            print(f"** Prompt: **\n{prompt}")
            print("*" * 50)
            print(f"** Completion: **\n{completion}")
            print("*" * 50)
            print("\n")
        elif EventPayload.MESSAGES in payload:
            messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
            messages_str = "\n".join([str(x) for x in messages])
            response = str(payload.get(EventPayload.RESPONSE))

            print(f"** Messages: **\n{messages_str}")
            print("*" * 50)
            print(f"** Response: **\n{response}")
            print("*" * 50)
            print("\n")

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
        """Count the LLM or Embedding tokens as needed."""
        if event_type == CBEventType.LLM and payload is not None:
            self._print_llm_event(payload)
