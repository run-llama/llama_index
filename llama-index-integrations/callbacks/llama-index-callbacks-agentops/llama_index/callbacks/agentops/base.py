import json
from inspect import BoundArguments
from typing import Any, Dict, List, Optional, Set

from agentops import Client as AOClient
from agentops import LLMEvent, ToolEvent, ErrorEvent

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.base_handler import BaseInstrumentationHandler
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.agent import (
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers.simple import SimpleSpanHandler
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr


class AgentOpsHandlerState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    is_agent_chat_span: Dict[str, bool] = Field(
        default_factory=dict,
        description="Dictionary to check whether a span originates from an agent.",
    )
    agent_chat_start_event: Dict[str, LLMChatStartEvent] = Field(
        default_factory=dict,
        description="Dictionary to hold a start event emitted by an agent.",
    )
    span_parent: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Dictionary to get parent span_id of a given span.",
    )
    span_exception: Dict[str, Set[Exception]] = Field(
        default_factory=dict,
        description="Dictionary to hold exceptions thrown in a span and its immediate children.",
    )

    def remove_span_id(self, span_id: str) -> None:
        """Remove a given span_id from all state fields."""
        self.is_agent_chat_span.pop(span_id, None)
        self.agent_chat_start_event.pop(span_id, None)
        self.span_parent.pop(span_id, None)
        self.span_exception.pop(span_id, None)

    def check_is_agent_chat_span(self, span_id: Optional[str]) -> bool:
        """
        Starting with a given span_id, navigate all ancestor spans to determine
        whether an AgentRunStepStartEvent is associated with at least one ancestor.
        """
        if not span_id:
            return False
        elif span_id in self.is_agent_chat_span and self.is_agent_chat_span[span_id]:
            return True
        else:
            return self.check_is_agent_chat_span(self.span_parent.get(span_id, None))

    def get_chat_start_event(
        self, span_id: Optional[str]
    ) -> Optional[LLMChatStartEvent]:
        """
        Starting with a given span_id, find the first ancestor span with an
        associated LLMChatStartEvent, then return this event.
        """
        if not span_id:
            return None
        elif span_id in self.agent_chat_start_event:
            return self.agent_chat_start_event[span_id]
        else:
            return self.get_chat_start_event(self.span_parent.get(span_id, None))


class AgentOpsSpanHandler(SimpleSpanHandler):
    _shared_handler_state: AgentOpsHandlerState = PrivateAttr()
    _ao_client: AOClient = PrivateAttr()
    _observed_exceptions: Set[Exception] = PrivateAttr()

    def __init__(
        self, shared_handler_state: AgentOpsHandlerState, ao_client: AOClient
    ) -> None:
        super().__init__()
        self._shared_handler_state = shared_handler_state
        self._ao_client = ao_client
        self._observed_exceptions = set()

    @classmethod
    def class_name(cls) -> str:
        return "AgentOpsSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        self._shared_handler_state.is_agent_chat_span[id_] = False
        self._shared_handler_state.span_parent[id_] = parent_span_id
        return super().new_span(
            id_, bound_args, instance, parent_span_id, tags, **kwargs
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        self._shared_handler_state.remove_span_id(id_)
        return super().prepare_to_exit_span(id_, bound_args, instance, result, **kwargs)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        if err:
            # Associate this exception to the parent span, so that it will know to ignore it
            parent_span_id = self._shared_handler_state.span_parent.get(id_, None)
            if parent_span_id:
                if parent_span_id in self._shared_handler_state.span_exception:
                    self._shared_handler_state.span_exception[parent_span_id].add(err)
                else:
                    self._shared_handler_state.span_exception[parent_span_id] = {err}

            # If this exception hasn't yet been associated to this span, record it
            if (
                id_ not in self._shared_handler_state.span_exception
                or err not in self._shared_handler_state.span_exception[id_]
            ):
                self._ao_client.record(ErrorEvent(details=str(err)))

        self._shared_handler_state.remove_span_id(id_)
        return super().prepare_to_drop_span(id_, bound_args, instance, err, **kwargs)


class AgentOpsEventHandler(BaseEventHandler):
    _shared_handler_state: AgentOpsHandlerState = PrivateAttr()
    _ao_client: AOClient = PrivateAttr()

    def __init__(
        self, shared_handler_state: AgentOpsHandlerState, ao_client: AOClient
    ) -> None:
        super().__init__()
        self._shared_handler_state = shared_handler_state
        self._ao_client = ao_client

    @classmethod
    def class_name(cls) -> str:
        return "AgentOpsEventHandler"

    def handle(self, event: BaseEvent) -> None:
        # We only track chat events that are emitted while using an agent
        is_agent_chat_event = self._shared_handler_state.check_is_agent_chat_span(
            event.span_id
        )

        if isinstance(event, AgentRunStepStartEvent):
            self._shared_handler_state.is_agent_chat_span[event.span_id] = True

        if isinstance(event, LLMChatStartEvent) and is_agent_chat_event:
            self._shared_handler_state.agent_chat_start_event[event.span_id] = event
        elif isinstance(event, LLMChatEndEvent) and is_agent_chat_event:
            message_dicts = []
            for message in event.messages:
                message_dicts.append(
                    {
                        "content": message.content,
                        "role": message.role,
                    }
                )

            result_dict = None
            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            if event.response:
                result_dict = {
                    "content": event.response.message.content,
                    "role": event.response.message.role,
                }
                if event.response.raw:
                    usage = dict(event.response.raw.get("usage", {}))
                    completion_tokens = usage.get("completion_tokens", None)
                    prompt_tokens = usage.get("prompt_tokens", None)
                    usage["prompt_tokens"] = prompt_tokens
                    usage["completion_tokens"] = completion_tokens

            event_params: Dict[str, Any] = {
                "prompt": message_dicts,
                "completion": result_dict,
                **usage,
            }

            # Get model info from chat start event corresponding to this chat end event
            start_event = self._shared_handler_state.get_chat_start_event(event.span_id)
            if start_event:
                event_params["model"] = (
                    start_event.model_dict["model"]
                    if "model" in start_event.model_dict
                    else None
                )

            self._ao_client.record(LLMEvent(**event_params))

        elif isinstance(event, AgentToolCallEvent):
            params = json.loads(event.arguments) if event.arguments else None
            self._ao_client.record(ToolEvent(name=event.tool.name, params=params))


class AgentOpsHandler(BaseInstrumentationHandler):
    @classmethod
    def init(
        cls,
        api_key: Optional[str] = None,
        parent_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_wait_time: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        instrument_llm_calls=True,
        inherited_session_id: Optional[str] = None,
    ):
        client_params: Dict[str, Any] = {
            "api_key": api_key,
            "parent_key": parent_key,
            "endpoint": endpoint,
            "max_wait_time": max_wait_time,
            "max_queue_size": max_queue_size,
            "tags": tags,
            "instrument_llm_calls": instrument_llm_calls,
            "auto_start_session": True,
            "inherited_session_id": inherited_session_id,
            "skip_auto_end_session": False,
        }
        ao_client = AOClient(
            **{k: v for k, v in client_params.items() if v is not None}
        )

        # Create synchronized span and event handler, attach to root dispatcher
        dispatcher = instrument.get_dispatcher()
        handler_state = AgentOpsHandlerState()
        event_handler = AgentOpsEventHandler(
            shared_handler_state=handler_state, ao_client=ao_client
        )
        span_handler = AgentOpsSpanHandler(
            shared_handler_state=handler_state, ao_client=ao_client
        )
        dispatcher.add_event_handler(event_handler)
        dispatcher.add_span_handler(span_handler)
