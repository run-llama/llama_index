import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload


class OpenAIFineTuningHandler(BaseCallbackHandler):
    """
    Callback handler for OpenAI fine-tuning.

    This handler will collect all messages
    sent to the LLM, along with their responses. It will then save these messages
    in a `.jsonl` format that can be used for fine-tuning with OpenAI's API.
    """

    def __init__(self, output_cls: Optional[Type[BaseModel]] = None) -> None:
        """Initialize the base callback handler."""
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._finetuning_events: Dict[str, List[Any]] = {}
        self._function_calls: Dict[str, List[Any]] = {}
        self._output_cls = output_cls

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
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

            # if functions exists, add that
            if payload and EventPayload.ADDITIONAL_KWARGS in payload:
                kwargs_dict = payload[EventPayload.ADDITIONAL_KWARGS]
                if "functions" in kwargs_dict:
                    self._function_calls[event_id] = kwargs_dict["functions"]
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

    def get_finetuning_events(self) -> Dict[str, Dict[str, Any]]:
        events_dict = {}
        for event_id, event in self._finetuning_events.items():
            events_dict[event_id] = {"messages": event[:-1], "response": event[-1]}

        return events_dict

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

        events_dict = self.get_finetuning_events()
        json_strs = []
        for event_id, event in events_dict.items():
            all_messages = event["messages"] + [event["response"]]
            message_dicts = to_openai_message_dicts(all_messages, drop_none=True)
            event_dict = {"messages": message_dicts}
            if event_id in self._function_calls:
                event_dict["functions"] = self._function_calls[event_id]
            json_strs.append(json.dumps(event_dict))

        with open(path, "w") as f:
            f.write("\n".join(json_strs))
        print(f"Wrote {len(json_strs)} examples to {path}")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
