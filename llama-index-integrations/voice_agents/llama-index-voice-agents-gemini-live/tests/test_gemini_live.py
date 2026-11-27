from unittest.mock import MagicMock, patch
from llama_index.voice_agents.gemini_live.base import GeminiLiveVoiceAgent


def test_client_header_initialization():
    """Test that the client header is correctly passed to the GeminiLiveVoiceAgent."""
    with patch("llama_index.voice_agents.gemini_live.base.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        agent = GeminiLiveVoiceAgent(api_key="test-key")

        # Access the client property to trigger initialization
        _ = agent.client

        # Check if http_options were passed to the client constructor
        call_args = mock_client_class.call_args
        _, kwargs = call_args

        http_options = kwargs["http_options"]
        headers = http_options["headers"]
        assert headers["x-goog-api-client"].startswith("llamaindex/")


def test_default_model_is_not_deprecated_preview():
    """The default model should be Google's named replacement for the deprecated native-audio preview."""
    agent = GeminiLiveVoiceAgent(api_key="test-key")
    assert agent.model == "gemini-3.8-live"


def test_explicit_model_is_kept():
    agent = GeminiLiveVoiceAgent(api_key="test-key", model="my-custom-model")
    assert agent.model == "my-custom-model"
