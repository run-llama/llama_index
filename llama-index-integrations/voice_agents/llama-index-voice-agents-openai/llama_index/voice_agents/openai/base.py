import base64
import os
import threading

from typing import List, Optional, Union
from .types import (
    ConversationStartEvent,
    ConversationInputEvent,
    ConversationDeltaEvent,
    ConversationDoneEvent,
    ConversationBaseEvent,
    ConversationResponse,
)
from .audio_interface import ConversationAudioInterface
from .websocket import ConversationWebSocket
from llama_index.core.llms import ChatMessage, MessageRole, AudioBlock, TextBlock


class OpenAIConversation:
    """
    Interface for the OpenAI Realtime Conversation integration with LlamaIndex.

    Attributes:
        api_key (Optional[str]): the OpenAI API key. Defaults to the environmental variable OPENAI_API_KEY if the value is None.
        ws_url (str): the URL for the OpenAI Realtime Conversation websocket. Defaults to: 'wss://api.openai.com/v1/realtime'
        model (str): the conversational model. Defaults to: 'gpt-4o-realtime-preview-2024-10-01'

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ws_url: str = "wss://api.openai.com/v1/realtime",
        model: str = "gpt-4o-realtime-preview-2024-10-01",
    ) -> None:
        url = ws_url + "?model=" + model
        openai_api_key = os.getenv("OPENAI_API_KEY", None) or api_key
        self.socket = ConversationWebSocket(
            openai_api_key, url, on_msg=self.handle_message
        )
        self.audio_io = ConversationAudioInterface(
            on_audio_callback=self.send_audio_to_socket
        )
        self.recv_thread: Optional[threading.Thread] = None
        self._all_events: List[ConversationBaseEvent] = []
        self._all_messages: List[ChatMessage] = []

    def start(self, instructions: str = "Please assist the user.") -> None:
        """
        Start the conversation and all related processes.

        Args:
            instructions (str): The system instructions for the assistant to start the conversation.

        """
        self.socket.connect()

        start_event = ConversationStartEvent(
            type_t="response.create",
            response=ConversationResponse(instructions=instructions),
        )
        self._all_events.append(start_event)
        self._all_messages.append(ChatMessage(role="system", content=instructions))
        # Send initial request to start the conversation
        self.socket.send(start_event.model_dump(by_alias=True))

        # Start processing microphone audio
        self.audio_thread = threading.Thread(target=self.audio_io.process_mic_audio)
        self.audio_thread.start()

        # Start audio streams (mic and speaker)
        self.audio_io.start_streams()

    def send_audio_to_socket(self, mic_chunk: bytes) -> None:
        """
        Callback function to send audio data to the OpenAI Conversation Websocket.

        Args:
            mic_chunk (bytes): the incoming audio stream from the user's input device.

        """
        encoded_chunk = base64.b64encode(mic_chunk).decode("utf-8")
        audio_event = ConversationInputEvent(
            type_t="input_audio_buffer.append", audio=encoded_chunk
        )
        self._all_events.append(audio_event)
        self._all_messages.append(
            ChatMessage(role=MessageRole.USER, blocks=[AudioBlock(audio=mic_chunk)])
        )
        self.socket.send(audio_event.model_dump(by_alias=True))

    def handle_message(self, message: dict) -> None:
        """
        Handle incoming message from OpenAI Conversation Websocket.

        Args:
            message (dict): The message from the websocket.

        """
        message["type_t"] = message.pop("type")
        if message["type_t"] == "response.audio.delta":
            event: ConversationBaseEvent = ConversationDeltaEvent.model_validate(
                message
            )
            audio_content = base64.b64decode(message["delta"])
            self._all_messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=audio_content)]
                )
            )
            self.audio_io.receive_audio(audio_content)

        elif message["type_t"] == "response.text.delta":
            event = ConversationDeltaEvent.model_validate(message)
            self._all_messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[TextBlock(text=event.delta)]
                )
            )

        elif message["type_t"] == "response.audio_transcript.delta":
            event = ConversationDeltaEvent.model_validate(message)
            self._all_messages.append(
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
        else:
            return
        self._all_events.append(event)

    def stop(self) -> None:
        """
        Stop the conversation and close all the related processes.
        """
        # Signal threads to stop
        self.audio_io._stop_event.set()
        self.socket.kill()

        # Stop audio streams
        self.audio_io.stop_streams()

        # Join threads to ensure they exit cleanly
        if self.audio_thread:
            self.audio_thread.join()

    def export_all_events(
        self, as_dict: bool = False
    ) -> Union[List[ConversationBaseEvent], List[dict]]:
        """
        Export all events occurred during the conversation.

        Args:
            as_dict (bool): Export all the events not as Pydantic BaseModels, but as their serialized, dictionary representation. Defaults to False.

        """
        if as_dict:
            return [event.model_dump(by_alias=True) for event in self._all_events]
        return self._all_events

    def export_all_messages(self) -> List[ChatMessage]:
        return self._all_messages
