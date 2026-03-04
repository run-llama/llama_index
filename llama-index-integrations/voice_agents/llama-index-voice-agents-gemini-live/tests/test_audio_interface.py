import pytest
import asyncio

from llama_index.core.voice_agents.interface import BaseVoiceAgentInterface
from llama_index.voice_agents.gemini_live.audio_interface import (
    GeminiLiveVoiceAgentInterface,
)
from google.genai.live import AsyncSession
from typing_extensions import override


class MockSession(AsyncSession):
    @override
    def __init__(self):
        pass


@pytest.mark.asyncio
async def test_audio_interface():
    interface = GeminiLiveVoiceAgentInterface()
    assert isinstance(interface, BaseVoiceAgentInterface)
    assert interface.audio_in_queue is None
    assert interface.out_queue is None
    assert interface.session is None
    assert interface.audio_stream is None
    interface.start(session=MockSession())
    assert isinstance(interface.session, MockSession)
    assert isinstance(interface.out_queue, asyncio.Queue)
    assert isinstance(interface.audio_in_queue, asyncio.Queue)
    await interface.receive(data=b"hello world")
    assert await interface.audio_in_queue.get() == b"hello world"
