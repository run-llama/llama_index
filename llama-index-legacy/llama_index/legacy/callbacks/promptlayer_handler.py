import datetime
from typing import Any, Dict, List, Optional, Union, cast

from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.callbacks.base_handler import BaseCallbackHandler
from llama_index.legacy.callbacks.schema import CBEventType, EventPayload
from llama_index.legacy.llms import ChatMessage

PROMPT_LAYER_CHAT_FUNCTION_NAME = "llamaindex.chat.openai"
PROMPT_LAYER_COMPLETION_FUNCTION_NAME = "llamaindex.completion.openai"


class PromptLayerHandler(BaseCallbackHandler):
    """Callback handler for sending to promptlayer.com."""

    pl_tags: Optional[List[str]]
    return_pl_id: bool = False

    def __init__(self, pl_tags: List[str] = [], return_pl_id: bool = False) -> None:
        try:
            from promptlayer.utils import get_api_key, promptlayer_api_request

            self._promptlayer_api_request = promptlayer_api_request
            self._promptlayer_api_key = get_api_key()
        except ImportError:
            raise ImportError(
                "Please install PromptLAyer with `pip install promptlayer`"
            )
        self.pl_tags = pl_tags
        self.return_pl_id = return_pl_id
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    event_map: Dict[str, Dict[str, Any]] = {}

    def add_event(self, event_id: str, **kwargs: Any) -> None:
        self.event_map[event_id] = {
            "kwargs": kwargs,
            "request_start_time": datetime.datetime.now().timestamp(),
        }

    def get_event(
        self,
        event_id: str,
    ) -> Dict[str, Any]:
        return self.event_map[event_id] or {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.LLM and payload is not None:
            self.add_event(
                event_id=event_id, **payload.get(EventPayload.SERIALIZED, {})
            )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != CBEventType.LLM or payload is None:
            return
        request_end_time = datetime.datetime.now().timestamp()
        prompt = str(payload.get(EventPayload.PROMPT))
        completion = payload.get(EventPayload.COMPLETION)
        response = payload.get(EventPayload.RESPONSE)
        function_name = PROMPT_LAYER_CHAT_FUNCTION_NAME
        event_data = self.get_event(event_id=event_id)
        resp: Union[str, Dict]
        extra_args = {}
        if response:
            messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
            resp = response.message.dict()
            assert isinstance(resp, dict)

            usage_dict: Dict[str, int] = {}
            try:
                usage = response.raw.get("usage", None)  # type: ignore

                if isinstance(usage, dict):
                    usage_dict = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                elif isinstance(usage, BaseModel):
                    usage_dict = usage.dict()
            except Exception:
                pass

            extra_args = {
                "messages": [message.dict() for message in messages],
                "usage": usage_dict,
            }
            ## promptlayer needs tool_calls toplevel.
            if "tool_calls" in response.message.additional_kwargs:
                resp["tool_calls"] = [
                    tool_call.dict()
                    for tool_call in resp["additional_kwargs"]["tool_calls"]
                ]
                del resp["additional_kwargs"]["tool_calls"]
        if completion:
            function_name = PROMPT_LAYER_COMPLETION_FUNCTION_NAME
            resp = str(completion)
        pl_request_id = self._promptlayer_api_request(
            function_name,
            "openai",
            [prompt],
            {
                **extra_args,
                **event_data["kwargs"],
            },
            self.pl_tags,
            [resp],
            event_data["request_start_time"],
            request_end_time,
            self._promptlayer_api_key,
            return_pl_id=self.return_pl_id,
        )
