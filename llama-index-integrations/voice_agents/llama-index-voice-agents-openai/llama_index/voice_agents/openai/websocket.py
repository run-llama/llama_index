import threading
import json
import logging
import asyncio
import websockets

from websockets import ConnectionClosedError
from typing import Optional, Callable, Any
from llama_index.core.voice_agents import BaseVoiceAgentWebsocket

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAIVoiceAgentWebsocket(BaseVoiceAgentWebsocket):
    def __init__(
        self, uri: str, api_key: str, on_msg: Optional[Callable] = None
    ) -> None:
        super().__init__(uri=uri)
        self.api_key = api_key
        self.on_msg = on_msg
        self.send_queue: asyncio.Queue = asyncio.Queue()
        self._stop_event = threading.Event()
        self.loop_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def connect(self) -> None:
        """Start the socket loop in a new thread."""
        self.loop_thread = threading.Thread(target=self._run_socket_loop, daemon=True)
        self.loop_thread.start()

    async def aconnect(self) -> None:
        """Method not implemented."""
        raise NotImplementedError(
            f"This method has not been implemented for {self.__qualname__}"
        )

    def _run_socket_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._socket_loop())

    async def _socket_loop(self) -> None:
        """Establish connection and run send/recv loop."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            async with websockets.connect(self.uri, additional_headers=headers) as ws:
                self.ws = ws  # Safe: now created inside this thread + loop

                # Create separate tasks for sending and receiving
                recv_task = asyncio.create_task(self._recv_loop(ws))
                send_task = asyncio.create_task(self._send_loop(ws))

                try:
                    # Run both tasks concurrently until one completes or fails
                    await asyncio.gather(recv_task, send_task)
                except Exception as e:
                    logging.error(f"Error in socket tasks: {e}")
                finally:
                    # Clean up any remaining tasks
                    recv_task.cancel()
                    send_task.cancel()
                    await asyncio.gather(recv_task, send_task, return_exceptions=True)

        except Exception as e:
            logging.error(f"Failed to connect to WebSocket: {e}")

    async def _recv_loop(self, ws) -> None:
        """Handle incoming messages."""
        try:
            while not self._stop_event.is_set():
                try:
                    message = await ws.recv()
                    logging.info(f"Received message: {message}")
                    if message and self.on_msg:
                        await self.on_msg(json.loads(message))
                except ConnectionClosedError:
                    logging.error("WebSocket connection closed.")
                    break
        except Exception as e:
            logging.error(f"Error in receive loop: {e}")

    async def _send_loop(self, ws) -> None:
        """Handle outgoing messages."""
        try:
            while not self._stop_event.is_set():
                try:
                    # Wait for a message to send with a timeout to check stop_event
                    try:
                        message = await asyncio.wait_for(
                            self.send_queue.get(), timeout=0.1
                        )
                        await ws.send(json.dumps(message))
                    except asyncio.TimeoutError:
                        # Timeout is expected - just continue to check stop_event
                        continue
                except ConnectionClosedError:
                    logging.error("WebSocket connection closed.")
                    break
        except Exception as e:
            logging.error(f"Error in send loop: {e}")

    async def send(self, data: Any) -> None:
        """Enqueue a message for sending."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.send_queue.put_nowait, data)

    async def close(self) -> None:
        """Stop the loop and close the WebSocket."""
        self._stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread:
            self.loop_thread.join()
            logging.info("WebSocket loop thread terminated.")
