import pyaudio
import asyncio
from typing import Any, Optional
from typing_extensions import override

from google.genai.live import AsyncSession
from llama_index.core.voice_agents import BaseVoiceAgentInterface


FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()


class GeminiLiveVoiceAgentInterface(BaseVoiceAgentInterface):
    def __init__(self) -> None:
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        self.session: Optional[AsyncSession] = None
        self.audio_stream: Optional[pyaudio.Stream] = None

    def _speaker_callback(self, *args: Any, **kwargs: Any) -> Any:
        """
        Callback function for the audio output device.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """

    @override
    async def _microphone_callback(self) -> None:
        """
        Callback function for the audio input device.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    @override
    def start(self, session: AsyncSession) -> None:
        """
        Start the interface.

        Args:
            session (AsyncSession): the session to which the API is bound.

        """
        self.session = session
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)

    def stop(self) -> None:
        """
        Stop the interface.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        if self.audio_stream:
            self.audio_stream.close()
        else:
            raise ValueError("Audio stream has never been opened, cannot be closed.")

    def interrupt(self) -> None:
        """
        Interrupt the interface.

        Args:
            None
        Returns:
            out (None): This function does not return anything.

        """
        self.audio_in_queue.get_nowait()

    @override
    async def output(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process and output the audio.

        Args:
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    @override
    async def receive(self, data: bytes) -> Any:
        """
        Receive audio data.

        Args:
            data (Any): received audio data (generally as bytes or str, but it is kept open also to other types).
            *args: Can take any positional argument.
            **kwargs: Can take any keyword argument.

        Returns:
            out (Any): This function can return any output.

        """
        self.audio_in_queue.put_nowait(data)
