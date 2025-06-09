import threading
import queue
import json
import logging
import select

from typing import Optional, Callable, Any
from websocket import create_connection, WebSocketConnectionClosedException

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ConversationWebSocket:
    def __init__(
        self, api_key: str, ws_url: str, on_msg: Optional[Callable] = None
    ) -> None:
        self.api_key = api_key
        self.ws_url = ws_url
        self.ws = None
        self.on_msg = on_msg  # Callback for when a message is received
        self.send_queue: queue.Queue = queue.Queue()  # Outgoing message queue
        self._stop_event = threading.Event()
        self.loop_thread = None  # Store thread reference

    def connect(self) -> None:
        """Connect to WebSocket and start main loop."""
        self.ws = create_connection(
            self.ws_url,
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Beta: realtime=v1",
            ],
        )

        # Start a unified loop for sending and receiving messages
        self.loop_thread = threading.Thread(target=self._socket_loop)
        self.loop_thread.start()

    def _socket_loop(self) -> None:
        """Main loop that handles both sending and receiving messages."""
        while not self._stop_event.is_set():
            try:
                # Use select to check if WebSocket has data to read
                rlist, _, _ = select.select([self.ws.sock], [], [], 0.1)

                # If there's incoming data, receive it
                if rlist:
                    message = self.ws.recv()
                    if message and self.on_msg:
                        self.on_msg(
                            json.loads(message)
                        )  # Call the user-provided callback

                # Check if there's a message in the queue to send
                try:
                    outgoing_message = self.send_queue.get_nowait()
                    self.ws.send(json.dumps(outgoing_message))
                except queue.Empty:
                    continue  # No message to send, loop back
            except WebSocketConnectionClosedException:
                logging.error("WebSocket connection closed.")
                break
            except Exception as e:
                logging.error(f"Error in socket loop: {e}")
                break

    def send(self, data: Any) -> None:
        """Enqueue the message to be sent."""
        self.send_queue.put(data)

    def kill(self) -> None:
        """Cleanly shut down the WebSocket and stop the loop."""
        self._stop_event.set()

        # Close WebSocket
        if self.ws:
            try:
                self.ws.send_close()
                self.ws.close()
            except Exception as e:
                logging.error(f"Error closing WebSocket: {e}")

        # Ensure the loop thread is joined
        if self.loop_thread:
            self.loop_thread.join()
            logging.info("WebSocket loop thread terminated.")
