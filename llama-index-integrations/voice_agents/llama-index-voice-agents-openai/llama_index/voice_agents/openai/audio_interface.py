import pyaudio
import queue
import time
import logging
import threading

from typing import Callable, Optional, Any, Tuple, Union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

CHUNK_SIZE = 1024
RATE = 24000
FORMAT = pyaudio.paInt16
REENGAGE_DELAY_MS = 500


class ConversationAudioInterface:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        rate: int = RATE,
        format: int = FORMAT,
        on_audio_callback: Optional[Callable] = None,
    ):
        self.chunk_size = chunk_size
        self.rate = rate
        self.format = format
        self.audio_buffer = bytearray()
        self.mic_queue: queue.Queue = queue.Queue()
        self.mic_on_at: Union[int, float] = 0
        self.mic_active: Optional[bool] = None
        self._stop_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.on_audio_callback = on_audio_callback  # Callback for audio data

    def _mic_callback(
        self, in_data: Any, frame_count: int, time_info: Any, status: Any
    ) -> Tuple[None, Any]:
        """Microphone callback that queues audio chunks."""
        if time.time() > self.mic_on_at:
            if not self.mic_active:
                self.mic_active = True
            self.mic_queue.put(in_data)
        else:
            if self.mic_active:
                self.mic_active = False
        return (None, pyaudio.paContinue)

    def _spkr_callback(
        self, in_data: Any, frame_count: int, time_info: Any, status: Any
    ) -> Tuple[bytes, Any]:
        """Speaker callback that plays audio."""
        bytes_needed = frame_count * 2
        current_buffer_size = len(self.audio_buffer)

        if current_buffer_size >= bytes_needed:
            audio_chunk = bytes(self.audio_buffer[:bytes_needed])
            self.audio_buffer = self.audio_buffer[bytes_needed:]
            self.mic_on_at = time.time() + REENGAGE_DELAY_MS / 1000
        else:
            audio_chunk = bytes(self.audio_buffer) + b"\x00" * (
                bytes_needed - current_buffer_size
            )
            self.audio_buffer.clear()

        return (audio_chunk, pyaudio.paContinue)

    def start_streams(self) -> None:
        """Start microphone and speaker streams."""
        self.mic_stream = self.p.open(
            format=self.format,
            channels=1,
            rate=self.rate,
            input=True,
            stream_callback=self._mic_callback,
            frames_per_buffer=self.chunk_size,
        )
        self.spkr_stream = self.p.open(
            format=self.format,
            channels=1,
            rate=self.rate,
            output=True,
            stream_callback=self._spkr_callback,
            frames_per_buffer=self.chunk_size,
        )
        self.mic_stream.start_stream()
        self.spkr_stream.start_stream()

    def stop_streams(self) -> None:
        """Stop and close audio streams."""
        self.mic_stream.stop_stream()
        self.mic_stream.close()
        self.spkr_stream.stop_stream()
        self.spkr_stream.close()
        self.p.terminate()

    def process_mic_audio(self) -> None:
        """Process microphone audio and call back when new audio is ready."""
        while not self._stop_event.is_set():
            if not self.mic_queue.empty():
                mic_chunk = self.mic_queue.get()
                if self.on_audio_callback:
                    self.on_audio_callback(
                        mic_chunk
                    )  # Pass the audio chunk to the callback
            else:
                time.sleep(0.05)  # Avoid tight loop when no audio is available

    def receive_audio(self, audio_chunk: bytes) -> None:
        """Appends audio data to the buffer for playback."""
        self.audio_buffer.extend(audio_chunk)
