from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


class BaseVoiceAgentWebsocket(ABC):
    """
    Abstract base class for a voice agent websocket.

    Attributes:
        uri (str): URL of the websocket.
        ws (Optional[ClientConnection]): Private attribute, initialized as None, represents the websocket client.

    """

    def __init__(
        self,
        uri: str,
    ):
        self.uri = uri
        self.ws: Optional[ClientConnection] = None

    def connect(self) -> None:
        """
        Connect to the websocket.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """

    async def aconnect(self) -> None:
        """
        Asynchronously connect to the websocket.

        The implementation should be:

        ```
        self.ws = await websockets.connect(uri=self.uri)
        ```

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """

    @abstractmethod
    async def send(self, data: Any) -> None:
        """
        Send data to the websocket.

        Args:
            data (Any): Data to send to the websocket.

        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close the connection with the websocket.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        ...
