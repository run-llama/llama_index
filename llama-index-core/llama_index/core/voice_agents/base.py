from typing import Optional, Any, Callable, List
from abc import ABC, abstractmethod
from .websocket import BaseVoiceAgentWebsocket
from .interface import BaseVoiceAgentInterface
from .events import BaseVoiceAgentEvent

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool


class BaseVoiceAgent(ABC):
    """
    Abstract class that serves as base for any Voice Agent.

    Attributes:
        ws (BaseVoiceAgentWebSocket): The websocket underlying the agent and providing the voice service.
        interface (BaseVoiceAgentInterface): The audio input/output interface.
        api_key (Optional[str]): API key (if needed). Defaults to None.
        tools (Optional[List[BaseTool]]): List of tools for the agent to use (tool use should be adapted to the specific integration). Defaults to None.
        _messages (List[ChatMessage]): Private attribute initialized as an empty list of ChatMessage, it should be populated with chat messages as the conversation goes on.
        _events (List[BaseVoiceAgentEvent]): Private attribute initialized as an empty list of BaseVoiceAgentEvent, it should be populated with events as the conversation goes on.

    """

    def __init__(
        self,
        ws: Optional[BaseVoiceAgentWebsocket] = None,
        interface: Optional[BaseVoiceAgentInterface] = None,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        self.ws = ws
        self.ws_url = ws_url
        self.interface = interface
        self.api_key = api_key
        self.tools = tools
        self._messages: List[ChatMessage] = []
        self._events: List[BaseVoiceAgentEvent] = []

    @abstractmethod
    async def start(self, *args: Any, **kwargs: Any) -> None:
        """
        Start the voice agent.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    async def send(self, audio: Any, *args: Any, **kwargs: Any) -> None:
        """
        Send audio to the websocket underlying the voice agent.

        Args:
            audio (Any): audio data to send (generally as bytes or str, but it is kept open also to other types).
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    async def handle_message(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Handle incoming message.

        Args:
            message (Any): incoming message (should be dict, but it is kept open also for other types).
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        ...

    @abstractmethod
    async def interrupt(self) -> None:
        """
        Interrupt the input/output audio flow.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the conversation with the voice agent.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        ...

    def export_messages(
        self,
        limit: Optional[int] = None,
        filter: Optional[Callable[[List[ChatMessage]], List[ChatMessage]]] = None,
    ) -> List[ChatMessage]:
        """
        Export all recorded messages during a conversation.

        Args:
            limit (Optional[int]): Maximum number of messages to return. Defaults to None.
            filter (Optional[Callable[[List[ChatMessage]], List[ChatMessage]]]): Filter function. Defaults to None.

        Returns:
            out (List[ChatMessage]): exported messages.

        """
        messages = self._messages
        if limit:
            if limit <= len(messages):
                messages = messages[:limit]
        if filter:
            messages = filter(messages)
        return messages

    def export_events(
        self,
        limit: Optional[int] = None,
        filter: Optional[
            Callable[[List[BaseVoiceAgentEvent]], List[BaseVoiceAgentEvent]]
        ] = None,
    ) -> List[BaseVoiceAgentEvent]:
        """
        Export all recorded events during a conversation.

        Args:
            limit (Optional[int]): Maximum number of events to return. Defaults to None.
            filter (Optional[Callable[[List[BaseVoiceAgentEvent]], List[BaseVoiceAgentEvent]]]): Filter function. Defaults to None.

        Returns:
            out (List[BaseVoiceAgentEvent]): exported events.

        """
        events = self._events
        if limit:
            if limit <= len(events):
                events = events[:limit]
        if filter:
            events = filter(events)
        return events
