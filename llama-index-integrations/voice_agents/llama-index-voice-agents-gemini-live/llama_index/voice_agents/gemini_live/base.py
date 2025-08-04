import asyncio
import logging
from typing import Optional, Any, List, Dict, Callable
from typing_extensions import override

from .audio_interface import GeminiLiveVoiceAgentInterface
from .utils import tools_to_gemini_tools, tools_to_functions_dict
from .events import (
    TextReceivedEvent,
    AudioReceivedEvent,
    ToolCallEvent,
    ToolCallResultEvent,
)
from google.genai.live import AsyncSession
from google.genai import Client, types
from llama_index.core.llms import ChatMessage, AudioBlock, TextBlock
from llama_index.core.voice_agents import BaseVoiceAgent
from llama_index.core.tools import BaseTool

DEFAULT_MODEL = "models/gemini-2.0-flash-live-001"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
        self._quitflag: bool = False
        interface = interface or GeminiLiveVoiceAgentInterface()
        super().__init__(api_key=api_key, tools=tools, interface=interface)
        if self.tools is not None:
            self.gemini_tools: List[Dict[str, List[Dict[str, str]]]] = (
                tools_to_gemini_tools(tools)
            )
            self._functions_dict: Dict[
                str, Callable[[Dict[str, Any], str, str], types.FunctionResponse]
            ] = tools_to_functions_dict(self.tools)
        else:
            self.gemini_tools = []
            self._functions_dict = {}

    @property
    def client(self) -> Client:
        if not self._client:
            self._client = Client(
                api_key=self.api_key, http_options={"api_version": "v1beta"}
            )
        return self._client

    def _signal_exit(self):
        logging.info("Preparing exit...")
        self._quitflag = True

    @override
    async def _start(self, session: AsyncSession) -> None:
        """
        Start the voice agent.
        """
        self.interface.start(session=session)

    async def _run_loop(self) -> None:
        logging.info("The agent is ready for the conversation")
        logging.info("Type q and press enter to stop the conversation at any time")
        while not self._quitflag:
            text = await asyncio.to_thread(
                input,
                "",
            )
            if text == "q":
                self._signal_exit()
            await self.session.send(input=text or ".", end_of_turn=True)
        logging.info("Session has been successfully closed")
        await self.interrupt()
        await self.stop()

    async def send(self) -> None:
        """
        Send audio to the websocket underlying the voice agent.
        """
        while True:
            msg = await self.interface.out_queue.get()
            await self.session.send(input=msg)

    @override
    async def handle_message(self) -> Any:
        """
        Handle incoming message.

        Args:
            message (Any): incoming message (should be dict, but it is kept open also for other types).
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.server_content:
                    if data := response.data:
                        await self.interface.receive(data=data)
                        self._messages.append(
                            ChatMessage(
                                role="assistant", blocks=[AudioBlock(audio=data)]
                            )
                        )
                        self._events.append(
                            AudioReceivedEvent(type_t="audio_received", data=data)
                        )
                        continue
                    if text := response.text:
                        self._messages.append(
                            ChatMessage(role="assistant", blocks=[TextBlock(text=text)])
                        )
                        self._events.append(
                            TextReceivedEvent(type_t="text_received", text=text)
                        )
                elif tool_call := response.tool_call:
                    function_responses: List[types.FunctionResponse] = []
                    for fn_call in tool_call.function_calls:
                        self._events.append(
                            ToolCallEvent(
                                type_t="tool_call",
                                tool_name=fn_call.name,
                                tool_args=fn_call.args,
                            )
                        )
                        result = self._functions_dict[fn_call.name](
                            fn_call.args, fn_call.id, fn_call.name
                        )
                        self._events.append(
                            ToolCallResultEvent(
                                type_t="tool_call_result",
                                tool_name=result.name,
                                tool_result=result.response,
                            )
                        )
                        function_responses.append(result)
                    await self.session.send_tool_response(
                        function_responses=function_responses
                    )
            while not self.interface.audio_in_queue.empty():
                await self.interrupt()

    async def start(self):
        try:
            async with (
                self.client.aio.live.connect(
                    model=self.model,
                    config={
                        "response_modalities": ["AUDIO"],
                        "tools": self.gemini_tools,
                    },
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                await self._start(session=session)

                _run_loop = tg.create_task(self._run_loop())
                tg.create_task(self.send())
                tg.create_task(self.interface._microphone_callback())
                tg.create_task(self.handle_message())
                tg.create_task(self.interface.output())

                await _run_loop
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
