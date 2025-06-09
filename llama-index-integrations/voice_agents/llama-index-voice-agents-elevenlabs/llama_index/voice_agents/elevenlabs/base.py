import base64
import json
import threading
import warnings

from statistics import mean
from websockets.sync.client import Connection
from typing import Optional, Callable, Dict, List, Any, Union
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ClientTools,
    ConversationInitiationData,
    AudioInterface,
)
from elevenlabs.base_client import BaseElevenLabs
from .utils import (
    callback_agent_message,
    callback_agent_message_correction,
    callback_latency_measurement,
    callback_user_message,
    make_function_from_tool_model,
)


class ElevenLabsConversation(Conversation):
    """
    Conversational AI session.

    BETA: This API is subject to change without regard to backwards compatibility.

    Attributes:
        client (BaseElevenLabs): The ElevenLabs client to use for the conversation.
        agent_id (str): The ID of the agent to converse with.
        requires_auth (bool): Whether the agent requires authentication.
        audio_interface (AudioInterface): The audio interface to use for input and output.
        config (Optional[ConversationInitiationData]): The configuration for the conversation
        client_tools (Optional[ClientTools]): The client tools to use for the conversation.

    """

    client: BaseElevenLabs
    agent_id: str
    requires_auth: bool
    config: ConversationInitiationData
    audio_interface: AudioInterface
    tools: Optional[List[BaseTool]]

    _last_message_id: int
    _callback_agent_response: Callable
    _callback_agent_response_correction: Callable
    _callback_user_transcript: Callable
    _callback_latency_measurement: Callable
    _messages: Dict[int, ChatMessage]
    _thread: Optional[threading.Thread]
    _should_stop: threading.Event
    _conversation_id: Optional[str]
    _last_interrupt_id: int
    _ws: Optional[Connection]

    def __init__(
        self,
        client: BaseElevenLabs,
        agent_id: str,
        *,
        requires_auth: bool,
        audio_interface: AudioInterface,
        config: Optional[ConversationInitiationData] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.requires_auth = requires_auth
        self.audio_interface = audio_interface

        self.config = config or ConversationInitiationData()
        client_tools = ClientTools()
        if tools:
            for tool in tools:
                if tool.metadata.fn_schema is not None:
                    fn = make_function_from_tool_model(
                        model_cls=tool.metadata.fn_schema, tool=tool
                    )
                    client_tools.register(
                        tool_name=tool.metadata.get_name(), handler=fn
                    )
                else:
                    warnings.warn(
                        f"Tool {tool.metadata.get_name()} could not added, since its function schema seems to be unavailable"
                    )

        self.client_tools = client_tools or ClientTools()
        self.client_tools.start()

        self._callback_agent_response = callback_agent_message
        self._callback_agent_response_correction = callback_agent_message_correction
        self._callback_user_transcript = callback_user_message
        self._callback_latency_measurement = callback_latency_measurement
        self._latencies: List[int] = []
        self._all_chat: Dict[int, List[ChatMessage]] = {}
        self._current_message_id: int = 0
        self._thread = None
        self._ws: Optional[Connection] = None
        self._should_stop = threading.Event()
        self._conversation_id = None
        self._last_interrupt_id = 0

    def _handle_message(self, message: Dict, ws: Any) -> None:
        if message["type"] == "conversation_initiation_metadata":
            event = message["conversation_initiation_metadata_event"]
            assert self._conversation_id is None
            self._conversation_id = event["conversation_id"]

        elif message["type"] == "audio":
            event = message["audio_event"]
            if int(event["event_id"]) <= self._last_interrupt_id:
                return
            audio = base64.b64decode(event["audio_base_64"])
            self._callback_agent_response(
                messages=self._all_chat,
                message_id=self._current_message_id,
                audio=event["audio_base_64"],
            )
            self.audio_interface.output(audio)
        elif message["type"] == "agent_response":
            event = message["agent_response_event"]
            self._callback_agent_response(
                messages=self._all_chat,
                message_id=self._current_message_id,
                text=event["agent_response"].strip(),
            )
        elif message["type"] == "agent_response_correction":
            event = message["agent_response_correction_event"]
            self._callback_agent_response_correction(
                messages=self._all_chat,
                message_id=self._current_message_id,
                text=event["corrected_agent_response"].strip(),
            )
        elif message["type"] == "user_transcript":
            self._current_message_id += 1
            event = message["user_transcription_event"]
            self._callback_user_transcript(
                messages=self._all_chat,
                message_id=self._current_message_id,
                text=event["user_transcript"].strip(),
            )
        elif message["type"] == "interruption":
            event = message["interruption_event"]
            self._last_interrupt_id = int(event["event_id"])
            self.audio_interface.interrupt()
        elif message["type"] == "ping":
            event = message["ping_event"]
            ws.send(
                json.dumps(
                    {
                        "type": "pong",
                        "event_id": event["event_id"],
                    }
                )
            )
            if event["ping_ms"] is None:
                event["ping_ms"] = 0
            self._callback_latency_measurement(self._latencies, int(event["ping_ms"]))
        elif message["type"] == "client_tool_call":
            tool_call = message.get("client_tool_call", {})
            tool_name = tool_call.get("tool_name")
            parameters = {
                "tool_call_id": tool_call["tool_call_id"],
                **tool_call.get("parameters", {}),
            }

            def send_response(response):
                if not self._should_stop.is_set():
                    ws.send(json.dumps(response))

            self.client_tools.execute_tool(tool_name, parameters, send_response)
            message = f"Calling tool: {tool_name} with parameters: {parameters}"
            self._callback_agent_response(
                messages=self._all_chat,
                message_id=self._current_message_id,
                text=message,
            )

        else:
            pass  # Ignore all other message types.

    def get_messages(
        self,
        limit: Optional[int] = None,
        filter: Optional[Callable[[List[ChatMessage]], List[ChatMessage]]] = None,
    ) -> List[ChatMessage]:
        """
        Get the list of messages produced by the user and the agent in a LlamaIndex-compatible format.

        Args:
            limit (Optional[int]): limit the number of returned messages to a maximum.
            filter (Optional[Callable[[List[ChatMessage]], List[ChatMessage]]]): a function that filters the list of messages to return only the ones that respect certain conditions.

        """
        if len(self._all_chat) > 0:
            vals = list(self._all_chat.values())
            messages = [message for messages in vals for message in messages]
            if limit:
                if limit > len(messages):
                    warnings.warn(
                        message="The provided limit exceeds the length of the  current chat history",
                        category=UserWarning,
                    )
                    return messages
                else:
                    return messages[:limit]
            elif filter:
                return filter(messages)
        else:
            warnings.warn(
                message="There are no recorded messages for now", category=UserWarning
            )
            messages = []
        return messages

    def get_average_latency(self) -> Union[int, float]:
        """
        Get the average latency of your conversational agent.
        """
        if len(self._latencies) > 1:
            return mean(self._latencies)
        elif len(self._latencies) == 1:
            return self._latencies[0]
        else:
            warnings.warn(
                message="There are no recorded latencies", category=UserWarning
            )
            return 0
