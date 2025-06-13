from typing import Any
from abc import ABC, abstractmethod


class BaseVoiceAgentInterface(ABC):
    """
    Abstract base class for a voice agent audio input/output interface.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Please implement this method by initializing the class with arbitrary attributes."""
        ...

    @abstractmethod
    def _speaker_callback(self, *args, **kwargs) -> Any:
        """
        Callback function for the audio output device.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        ...

    @abstractmethod
    def _microphone_callback(self, *args, **kwargs) -> Any:
        """
        Callback function for the audio input device.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        ...

    @abstractmethod
    def start(self, *args, **kwargs) -> None:
        """
        Start the interface.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the interface.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    def interrupt(self) -> None:
        """
        Interrupt the interface.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        ...

    @abstractmethod
    def output(self, *args, **kwargs) -> Any:
        """
        Process and output the audio.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        ...

    @abstractmethod
    def receive(self, data: Any, *args, **kwargs) -> Any:
        """
        Receive audio data.

        Args:
            data (Any): received audio data (generally as bytes or str, but it is kept open also to other types).
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        ...
