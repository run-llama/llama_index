import asyncio
from typing import Optional, Any, List
from typing_extensions import override

from .audio_interface import GeminiLiveVoiceAgentInterface
from google.genai.live import AsyncSession
from google.genai import Client
from llama_index.core.voice_agents import BaseVoiceAgent
from llama_index.core.tools import BaseTool

DEFAULT_MODEL = "models/gemini-2.0-flash-live-001"


class GeminiLiveVoiceAgent(BaseVoiceAgent):
    """
    Gemini Live Voice Agent.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        interface: Optional[GeminiLiveVoiceAgentInterface] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        self.model: str = model or DEFAULT_MODEL
        self._client: Optional[Client] = None
        self.session: Optional[AsyncSession] = None
        interface = interface or GeminiLiveVoiceAgentInterface()
        super().__init__(api_key=api_key, tools=tools, interface=interface)

    @property
    def client(self) -> Client:
        if not self._client:
            self._client = Client(
                api_key=self.api_key, http_options={"api_version": "v1beta"}
            )
        return self._client

    @override
    async def start(self, session: AsyncSession) -> None:
        """
        Start the voice agent.
        """
        self.interface.start(session=session)

    @override
    async def send(self) -> None:
        """
        Send audio to the websocket underlying the voice agent.
        """
        while True:
            msg = await self.interface.out_queue.get()
            await self.session.send(input=msg)

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

    async def run(self):
        try:
            async with (
                self.client.aio.live.connect(
                    model=self.model, config={"response_modalities": ["AUDIO"]}
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                await self.start(session=session)

                tg.create_task(self.send())
                tg.create_task(self.interface._microphone_callback())
                tg.create_task(self.interface.receive())
                tg.create_task(self.interface.output())

                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            await self.stop()

    async def interrupt(self) -> None:
        """
        Interrupt the input/output audio flow.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        self.interface.interrupt()

    async def stop(self) -> None:
        """
        Stop the conversation with the voice agent.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        self.interface.stop()
