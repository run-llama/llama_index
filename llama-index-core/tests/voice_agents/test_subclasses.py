import pytest
from typing import List, Any, Union, Optional, Iterable, AsyncIterable
from llama_index.core.voice_agents import (
    BaseVoiceAgent,
    BaseVoiceAgentInterface,
    BaseVoiceAgentWebsocket,
    BaseVoiceAgentEvent,
)
from llama_index.core.llms import ChatMessage

# Try to import websockets dependency
try:
    from websockets import ClientConnection, ClientProtocol
    from websockets.uri import WebSocketURI

    websockets_available = True
except ImportError:
    websockets_available = False
    # Create dummy classes to prevent NameError when defining mocks
    ClientConnection = object
    ClientProtocol = object
    WebSocketURI = object


class MockVoiceAgentInterface(BaseVoiceAgentInterface):
    def __init__(self, name: str = "interface") -> None:
        self.name = name
        self._is_started = False
        self._is_stopped = False
        self._num_interrupted = 0
        self._received: List[bytes] = []

    def _speaker_callback(self) -> None:
        self.name += "."

    def _microphone_callback(self) -> None:
        self.name += ","

    def start(self) -> None:
        self._is_started = True

    def stop(self) -> None:
        self._is_stopped = True

    def interrupt(self) -> None:
        self._num_interrupted += 1

    def output(self) -> List[Any]:
        return [self.name, self._is_started, self._is_stopped, self._num_interrupted]

    def receive(self, data: bytes) -> None:
        self._received.append(data)


class MockConnection(ClientConnection):
    def __init__(self):
        if websockets_available:
            self.ws_uri = WebSocketURI(
                secure=True, host="localhost", port=2000, path="", query=""
            )
            self.protocol = ClientProtocol(uri=self.ws_uri)
        self._sent: List[
            Union[
                str,
                bytes,
                Iterable[Union[str, bytes]],
                AsyncIterable[Union[str, bytes]],
            ]
        ] = []
        self._received: List[str] = []
        self._is_closed: bool = False

    async def send(
        self,
        message: Union[
            str, bytes, Iterable[Union[str, bytes]], AsyncIterable[Union[str, bytes]]
        ],
        text: Optional[bool] = None,
    ) -> None:
        self._sent.append(message)

    async def recv(self, decode: Optional[bool] = None) -> Any:
        self._received.append("Received a message")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._is_closed = True


class MockVoiceAgentWebsocket(BaseVoiceAgentWebsocket):
    def __init__(self, uri: str, api_key: str):
        super().__init__(uri=uri)
        self.api_key = api_key

    async def aconnect(self) -> None:
        self.ws: ClientConnection = MockConnection()

    def connect(self) -> None:
        pass

    async def send(
        self,
        data: Union[
            str, bytes, Iterable[Union[str, bytes]], AsyncIterable[Union[str, bytes]]
        ],
    ) -> None:
        await self.ws.send(message=data)

    async def close(self) -> Any:
        await self.ws.close()


class MockVoiceAgent(BaseVoiceAgent):
    def __init__(
        self,
        ws: BaseVoiceAgentWebsocket,
        interface: BaseVoiceAgentInterface,
        api_key: Optional[str] = None,
    ):
        super().__init__(ws=ws, interface=interface, api_key=api_key)
        self._is_started = False
        self._sent: List[Any] = []
        self._handled: List[dict] = []
        self._is_stopped = False

    async def start(self, *args, **kwargs) -> None:
        self._is_started = True

    async def send(self, audio: Any, *args, **kwargs) -> None:
        self._sent.append(audio)

    async def interrupt(self) -> None:
        pass

    async def handle_message(self, message: dict) -> Any:
        self._handled.append(message)

    async def stop(self) -> None:
        self._is_stopped = True


@pytest.fixture()
def mock_interface() -> BaseVoiceAgentInterface:
    return MockVoiceAgentInterface()


@pytest.fixture()
def mock_websocket() -> BaseVoiceAgentWebsocket:
    return MockVoiceAgentWebsocket(
        uri="wss://my.mock.websocket:8000", api_key="fake-api-key"
    )


@pytest.fixture()
def mock_agent() -> BaseVoiceAgent:
    return MockVoiceAgent(
        ws=MockVoiceAgentWebsocket(
            uri="wss://my.mock.websocket:8000", api_key="fake-api-key"
        ),
        interface=MockVoiceAgentInterface(),
    )


@pytest.mark.skipif(not websockets_available, reason="websockets library not installed")
def test_interface_subclassing(mock_interface: MockVoiceAgentInterface):
    mock_interface.start()
    mock_interface._speaker_callback()
    mock_interface._microphone_callback()
    mock_interface.receive(data=b"hello world!")
    mock_interface.interrupt()
    mock_interface.stop()
    assert mock_interface.output() == ["interface.,", True, True, 1]
    assert mock_interface._received == [b"hello world!"]


@pytest.mark.asyncio
@pytest.mark.skipif(not websockets_available, reason="websockets library not installed")
async def test_websocket_subclassing(mock_websocket: MockVoiceAgentWebsocket):
    await mock_websocket.aconnect()
    assert isinstance(mock_websocket.ws, MockConnection)
    await mock_websocket.send(data="hello world")
    await mock_websocket.send(data=b"this is a test")
    assert mock_websocket.ws._sent == ["hello world", b"this is a test"]
    await mock_websocket.close()
    assert mock_websocket.ws._is_closed


@pytest.mark.asyncio
@pytest.mark.skipif(not websockets_available, reason="websockets library not installed")
async def test_agent_subclassing(mock_agent: MockVoiceAgent):
    await mock_agent.start()
    assert mock_agent._is_started
    await mock_agent.send(audio="Hello world")
    assert mock_agent._sent == ["Hello world"]
    await mock_agent.handle_message(message={"type": "text", "content": "content"})
    assert mock_agent._handled == [{"type": "text", "content": "content"}]
    mock_agent._events = [
        BaseVoiceAgentEvent(type_t="send"),
        BaseVoiceAgentEvent(type_t="text"),
    ]
    mock_agent._messages = [
        ChatMessage(role="user", content="Hello world"),
        ChatMessage(role="assistant", content="content"),
    ]

    def filter_events(events: List[BaseVoiceAgentEvent]):
        return [event for event in events if event.type_t == "send"]

    assert mock_agent.export_events() == [
        BaseVoiceAgentEvent(type_t="send"),
        BaseVoiceAgentEvent(type_t="text"),
    ]
    assert mock_agent.export_events(filter=filter_events) == [
        BaseVoiceAgentEvent(type_t="send")
    ]
    assert mock_agent.export_events(limit=1) == [BaseVoiceAgentEvent(type_t="send")]

    def filter_messages(messages: List[ChatMessage]):
        return [message for message in messages if message.role == "assistant"]

    assert mock_agent.export_messages() == [
        ChatMessage(role="user", content="Hello world"),
        ChatMessage(role="assistant", content="content"),
    ]
    assert mock_agent.export_messages(limit=1) == [
        ChatMessage(role="user", content="Hello world")
    ]
    assert mock_agent.export_messages(filter=filter_messages) == [
        ChatMessage(role="assistant", content="content")
    ]
    await mock_agent.stop()
    assert mock_agent._is_stopped
