import pytest
import assemblyai as aai


@pytest.fixture(autouse=True)
def mock_assemblyai_api_key(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "mock-test-key")
    original = aai.settings.api_key
    aai.settings.api_key = "mock-test-key"
    yield
    aai.settings.api_key = original
