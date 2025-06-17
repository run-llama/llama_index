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
                while not self._stop_event.is_set():
                    try:
                        # Receive
                        recv_task = asyncio.create_task(ws.recv())
                        send_task = asyncio.create_task(self.send_queue.get())
                        done, pending = await asyncio.wait(
                            [recv_task, send_task],
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=0.1,
                        )

                        if recv_task in done:
                            message = recv_task.result()
                            if message and self.on_msg:
                                await self.on_msg(json.loads(message))

                        if send_task in done:
                            message = send_task.result()
                            await ws.send(json.dumps(message))

                        for task in pending:
                            task.cancel()

                    except ConnectionClosedError:
                        logging.error("WebSocket connection closed.")
                        break
                    except Exception as e:
                        logging.error(f"Error in socket loop: {e}")
                        break

        except Exception as e:
            logging.error(f"Failed to connect to WebSocket: {e}")

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
