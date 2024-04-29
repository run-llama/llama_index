"""
The hub handles the moderation of inter-spoke communication. As the hub and spokes operate in isolated processes, sockets are employed to transmit messages between these processes. Consequently, a Socket class is defined for facilitating communication.
"""

import json


class Socket:
    def __init__(self, sock) -> None:
        self.sock = sock

    def send(self, msg):
        self.sock.sendall(msg)
        self.sock.sendall(b"\n")

    # The length parameter can be altered to fit the size of the message
    def recv(self, length=1024):
        buffer = ""
        while True:
            msg = self.sock.recv(length).decode("utf-8")
            if not msg:
                break
            buffer += msg

            if "\n" in buffer:
                # Split the buffer at the newline to process the complete message
                complete_msg, _, buffer = buffer.partition("\n")

                # Attempt to deserialize the JSON data
                try:
                    return json.loads(
                        complete_msg
                    )  # Return the deserialized dictionary
                except json.JSONDecodeError:
                    # Handle error if JSON is not well-formed
                    break  # Or handle error accordingly
        return None

    def close(self):
        self.sock.close()
