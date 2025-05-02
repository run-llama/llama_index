"""
The hub handles the moderation of inter-spoke communication. As the hub and spokes operate in isolated processes, sockets are employed to transmit messages between these processes. Consequently, a Socket class is defined for facilitating communication.
"""

import json


class Socket:
    """
    A class to facilitate communication between isolated processes using sockets.

    Attributes:
        sock (socket.socket): The socket object used for communication.

    """

    def __init__(self, sock) -> None:
        """
        Initialize the Socket with a given socket object.

        Args:
            sock (socket.socket): The socket object used for communication.

        """
        self.sock = sock

    def send(self, msg):
        """
        Send a message through the socket.

        Args:
            msg (bytes): The message to be sent.

        """
        self.sock.sendall(msg)
        self.sock.sendall(b"\n")

    def recv(self, length=1024):
        """
        Receive a message from the socket.

        Args:
            length (int, optional): The maximum amount of data to be received at once. Default is 1024.

        Returns:
            dict: The deserialized JSON message received, or None if the message is not well-formed.

        """
        # The length parameter can be altered to fit the size of the message
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
        """
        Close the socket.
        """
        self.sock.close()
