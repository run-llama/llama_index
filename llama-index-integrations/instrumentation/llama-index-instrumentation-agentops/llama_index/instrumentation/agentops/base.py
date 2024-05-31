from inspect import BoundArguments
from typing import Any, Dict, List, Optional
from agentops import Client as AOClient
from agentops import LLMEvent, ToolEvent, ErrorEvent
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
from llama_index.core.instrumentation.events.span import SpanDropEvent
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers.simple import SimpleSpanHandler
from llama_index.core.llms.chatml_utils import completion_to_prompt, messages_to_prompt


class AgentOpsHandlerState:
    is_agent_chat_span: Dict[str, bool] = {}
    agent_chat_start_event: Dict[str, LLMChatStartEvent] = {}
    span_parent: Dict[str, Optional[str]] = {}


class AgentOpsSpanHandler(SimpleSpanHandler):
    def __init__(self) -> None:
        self.is_agent_chat_span = AgentOpsHandlerState.is_agent_chat_span
        self.agent_chat_start_event = AgentOpsHandlerState.agent_chat_start_event
        self.span_parent = AgentOpsHandlerState.span_parent
        return super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "AgentOpsSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any
    ) -> SimpleSpan:
        self.is_agent_chat_span[id_] = False
        self.span_parent[id_] = parent_span_id
        return super().new_span(id_, bound_args, instance, parent_span_id, **kwargs)

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any
    ) -> SimpleSpan:
        del self.is_agent_chat_span[id_]
        del self.agent_chat_start_event[id_]
        del self.span_parent[id_]
        return super().prepare_to_exit_span(id_, bound_args, instance, result, **kwargs)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any
    ) -> SimpleSpan:
        del self.is_agent_chat_span[id_]
        del self.agent_chat_start_event[id_]
        del self.span_parent[id_]
        return super().prepare_to_drop_span(id_, bound_args, instance, err, **kwargs)


class AgentOpsEventHandler(BaseEventHandler):
    def __init__(
        self,
        api_key: Optional[str] = None,
        parent_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_wait_time: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        instrument_llm_calls=True,
        auto_start_session=True,
        inherited_session_id: Optional[str] = None,
    ):
        self.is_agent_chat_span = AgentOpsHandlerState.is_agent_chat_span
        self.agent_chat_start_event = AgentOpsHandlerState.agent_chat_start_event
        self.span_parent = AgentOpsHandlerState.span_parent
        client_params: Dict[str, Any] = {
            "api_key": api_key,
            "parent_key": parent_key,
            "endpoint": endpoint,
            "max_wait_time": max_wait_time,
            "max_queue_size": max_queue_size,
            "tags": tags,
            "instrument_llm_calls": instrument_llm_calls,
            "auto_start_session": auto_start_session,
            "inherited_session_id": inherited_session_id,
        }
        self.ao_client = AOClient(
            **{k: v for k, v in client_params.items() if v is not None}
        )

    @classmethod
    def class_name(cls) -> str:
        return "AgentOpsEventHandler"

    def _is_agent_chat_span(self, span_id: Optional[str]) -> bool:
        """
        Starting with a given span_id, navigate all ancestor spans to determine
        whether an AgentRunStepStartEvent is associated with at least one ancestor.
        """
        if not span_id:
            return False
        elif span_id in self.is_agent_chat_span and self.is_agent_chat_span[span_id]:
            return True
        else:
            return self._is_agent_chat_span(self.span_parent.get(span_id, None))

    def _get_chat_start_event(
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
            return self._get_chat_start_event(self.span_parent.get(span_id, None))

    def handle(self, event: BaseEvent) -> None:
        # We only track chat events that are emitted while using an agent
        is_agent_chat_event = self._is_agent_chat_span(event.span_id)

        if isinstance(event, AgentRunStepStartEvent):
            self.is_agent_chat_span[event.span_id] = True

        if isinstance(event, LLMChatStartEvent) and is_agent_chat_event:
            self.agent_chat_start_event[event.span_id] = event
            model = event.model_dict["model"]
            prompt = messages_to_prompt(event.messages)
            self.ao_client.record(LLMEvent(model=model, prompt=prompt))

        elif isinstance(event, LLMChatEndEvent) and is_agent_chat_event:
            event_params: Dict[str, Any] = {
                "prompt": messages_to_prompt(event.messages),
                "completion": completion_to_prompt(
                    event.response.message.content if event.response else None
                ),
            }

            # Get model info from chat start event corresponding to this chat end event
            start_event = self._get_chat_start_event(event.span_id)
            if start_event:
                event_params["model"] = start_event.model_dict["model"]

            self.ao_client.record(LLMEvent(**event_params))

        elif isinstance(event, AgentToolCallEvent):
            self.ao_client.record(ToolEvent(name=event.tool.name))

        elif isinstance(event, SpanDropEvent):
            self.ao_client.record(ErrorEvent(details=event.err_str))
