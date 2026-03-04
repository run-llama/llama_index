from llama_index.core.voice_agents.interface import BaseVoiceAgentInterface
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface


class ElevenLabsVoiceAgentInterface(DefaultAudioInterface, BaseVoiceAgentInterface):
    def __init__(self, *args, **kwargs):
        super().__init__()

    # Some methods from BaseVoiceAgentInterface are not implemented in DefaultAudioInterface, so we implement toy methods here
    def _speaker_callback(self, *args, **kwargs):
        pass

    def _microphone_callback(self, *args, **kwargs):
        pass

    def receive(self, data, *args, **kwargs):
        pass
