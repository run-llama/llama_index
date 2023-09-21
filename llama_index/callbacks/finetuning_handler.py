import json
from typing import Any, Dict, List, Optional

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload


class OpenAIFineTuningHandler(BaseCallbackHandler):
    """
    Callback handler for OpenAI fine-tuning.

    This handler will collect all messages
    sent to the LLM, along with their responses. It will then save these messages
    in a `.jsonl` format that can be used for fine-tuning with OpenAI's API.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the base callback handler."""
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._finetuning_events: Dict[str, List[Any]] = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        from llama_index.llms.base import ChatMessage, MessageRole

        if event_type == CBEventType.LLM:
            cur_messages = []
            if payload and EventPayload.PROMPT in payload:
                message = ChatMessage(
                    role=MessageRole.USER, text=str(payload[EventPayload.PROMPT])
                )
                cur_messages = [message]
            elif payload and EventPayload.MESSAGES in payload:
                cur_messages = payload[EventPayload.MESSAGES]

            if len(cur_messages) > 0:
                if event_id in self._finetuning_events:
                    self._finetuning_events[event_id].extend(cur_messages)
                else:
                    self._finetuning_events[event_id] = cur_messages
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        from llama_index.llms.base import ChatMessage, MessageRole

        if (
            event_type == CBEventType.LLM
            and event_id in self._finetuning_events
            and payload is not None
        ):
            if isinstance(payload[EventPayload.RESPONSE], str):
                response = ChatMessage(
                    role=MessageRole.ASSISTANT, text=str(payload[EventPayload.RESPONSE])
                )
            else:
                response = payload[EventPayload.RESPONSE].message

            self._finetuning_events[event_id].append(response)

        return None

    def get_finetuning_events(self) -> List[Dict[str, Any]]:
        events = []
        for event in self._finetuning_events.values():
            events.append({"messages": event[:-1], "response": event[-1]})

        return events

    def save_finetuning_events(self, path: str) -> None:
        """
        Save the finetuning events to a file.

        This saved format can be used for fine-tuning with OpenAI's API.
        The structure for each json line is as follows:
        {
          messages: [
            { rol: "system", content: "Text"},
            { role: "user", content: "Text" },
          ]
        },
        ...
        """
        from llama_index.llms.openai_utils import to_openai_message_dicts

        events = self.get_finetuning_events()
        json_strs = []
        for event in events:
            all_messages = event["messages"] + [event["response"]]
            message_dicts = to_openai_message_dicts(all_messages)
            json_strs.append(json.dumps({"messages": message_dicts}))

        with open(path, "w") as f:
            f.write("\n".join(json_strs))
        print(f"Wrote {len(json_strs)} examples to {path}")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass
