import base64
import logging
import os
import threading

from typing import List, Optional, Dict, Any
from .types import (
    ConversationInputEvent,
    ConversationDeltaEvent,
    ConversationDoneEvent,
    ConversationSessionUpdate,
    ConversationSession,
    ConversationTool,
    ToolParameters,
)
from .audio_interface import OpenAIVoiceAgentInterface
from .websocket import OpenAIVoiceAgentWebsocket
from llama_index.core.llms import ChatMessage, MessageRole, AudioBlock, TextBlock
from llama_index.core.tools import BaseTool
from llama_index.core.voice_agents import (
    BaseVoiceAgent,
    BaseVoiceAgentEvent,
    BaseVoiceAgentInterface,
    BaseVoiceAgentWebsocket,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://api.openai.com/v1/realtime"
DEFALT_MODEL = "gpt-4o-realtime-preview"


class OpenAIVoiceAgent(BaseVoiceAgent):
    """
    >**NOTE**: *This API is a BETA, and thus might be subject to changes*.

    Interface for the OpenAI Realtime Conversation integration with LlamaIndex.

    Attributes:
        ws (Optional[BaseVoiceAgentWebsocket]): A pre-defined websocket to use. Defaults to None. In case of doubt, it is advised to leave this argument as None and pass ws_url and model.
        interface (Optional[BaseVoiceAgentInterface]): Audio I/O interface. Defaults to None. In case of doubt, it is advised to leave this argument as None.
        api_key (Optional[str]): The OpenAI API key. Defaults to the environmental variable OPENAI_API_KEY if the value is None.
        ws_url (str): The URL for the OpenAI Realtime Conversation websocket. Defaults to: 'wss://api.openai.com/v1/realtime'.
        model (str): The conversational model. Defaults to: 'gpt-4o-realtime-preview'.
        tools (List[BaseTool]): Tools to equip the agent with.

    """

    def __init__(
        self,
        ws: Optional[BaseVoiceAgentWebsocket] = None,
        interface: Optional[BaseVoiceAgentInterface] = None,
        api_key: Optional[str] = None,
        ws_url: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        super().__init__(
            ws=ws, interface=interface, ws_url=ws_url, api_key=api_key, tools=tools
        )
        if not self.ws:
            if not model:
                model = DEFALT_MODEL
            if not self.ws_url:
                self.ws_url = DEFAULT_WS_URL
            url = self.ws_url + "?model=" + model
            openai_api_key = os.getenv("OPENAI_API_KEY", None) or self.api_key
            if not openai_api_key:
                raise ValueError(
                    "The OPENAI_API_KEY is neither passed from the function arguments nor from environmental variables"
                )
            self.ws: OpenAIVoiceAgentWebsocket = OpenAIVoiceAgentWebsocket(
                uri=url, api_key=openai_api_key, on_msg=self.handle_message
            )
        if not self.interface:
            self.interface: OpenAIVoiceAgentInterface = OpenAIVoiceAgentInterface(
                on_audio_callback=self.send
            )
        self.recv_thread: Optional[threading.Thread] = None

    async def start(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Start the conversation and all related processes.

        Args:
            **kwargs (Any): You can pass all the keyword arguments related to initializing a session, except for `tools`, which is inferred from the `tools` attribute of the class. Find a reference for these arguments and their type [on OpenAI official documentation](https://platform.openai.com/docs/api-reference/realtime-client-events/session/update).

        """
        self.ws.connect()
        breakpoint()
        session = ConversationSession.model_validate(kwargs)
        logger.info(f"Session: {session}")

        if self.tools is not None:
            openai_conv_tools: List[ConversationTool] = []

            for tool in self.tools:
                params_dict = tool.metadata.get_parameters_dict()
                tool_params = ToolParameters.model_validate(params_dict)
                conv_tool = ConversationTool(
                    name=tool.metadata.get_name(),
                    description=tool.metadata.description,
                    parameters=tool_params,
                )
                openai_conv_tools.append(conv_tool)

            session.tools = openai_conv_tools

        update_session_event = ConversationSessionUpdate(
            type_t="session.update",
            session=session,
        )
        self._events.append(update_session_event)
        self._messages.append(ChatMessage(role="system", content=session.instructions))
        # Send initial request to start the conversation
        await self.ws.send(update_session_event.model_dump(by_alias=True))

        # Start processing microphone audio
        self.audio_thread = threading.Thread(target=self.interface.output)
        self.audio_thread.start()

        # Start audio streams (mic and speaker)
        self.interface.start()
        print("The agent is ready to have a conversation")

    async def send(self, audio: bytes, *args: Any, **kwargs: Any) -> None:
        """
        Callback function to send audio data to the OpenAI Conversation Websocket.

        Args:
            mic_chunk (bytes): the incoming audio stream from the user's input device.

        """
        encoded_chunk = base64.b64encode(audio).decode("utf-8")
        audio_event = ConversationInputEvent(
            type_t="input_audio_buffer.append", audio=encoded_chunk
        )
        self._events.append(audio_event)
        self._messages.append(
            ChatMessage(role=MessageRole.USER, blocks=[AudioBlock(audio=audio)])
        )
        await self.ws.send(audio_event.model_dump(by_alias=True))

    async def handle_message(self, message: dict, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming message from OpenAI Conversation Websocket.

        Args:
            message (dict): The message from the websocket.

        """
        message["type_t"] = message.pop("type")
        if message["type_t"] == "response.audio.delta":
            event: BaseVoiceAgentEvent = ConversationDeltaEvent.model_validate(message)
            audio_content = base64.b64decode(message["delta"])
            self._messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=audio_content)]
                )
            )
            self.interface.receive(audio_content)

        elif message["type_t"] == "response.text.delta":
            event = ConversationDeltaEvent.model_validate(message)
            self._messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[TextBlock(text=event.delta)]
                )
            )

        elif message["type_t"] == "response.audio_transcript.delta":
            event = ConversationDeltaEvent.model_validate(message)
            self._messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[TextBlock(text=event.delta)]
                )
            )

        elif message["type_t"] == "response.text.done":
            event = ConversationDoneEvent.model_validate(message)

        elif message["type_t"] == "response.audio_transcript.done":
            event = ConversationDoneEvent.model_validate(message)

        elif message["type_t"] == "response.audio.done":
            event = ConversationDoneEvent.model_validate(message)
        elif message["type_t"] == "error":
            logging.error(f"Error: {message['error']}")
        else:
            return
        self._events.append(event)

    async def stop(self) -> None:
        """
        Stop the conversation and close all the related processes.
        """
        # Signal threads to stop
        self.interface._stop_event.set()
        await self.ws.close()

        # Stop audio streams
        self.interface.stop()

        # Join threads to ensure they exit cleanly
        if self.audio_thread:
            self.audio_thread.join()

    async def interrupt(self) -> None:
        """
        Interrupts the input/output audio streaming.
        """
        self.interface.interrupt()
