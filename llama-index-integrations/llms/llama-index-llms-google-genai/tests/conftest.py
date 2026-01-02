import pytest
from unittest.mock import create_autospec, MagicMock, AsyncMock
import google.genai

from llama_index.llms.google_genai.files import FileManager
from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.conversion.responses import ResponseConverter
from llama_index.llms.google_genai.conversion.tools import ToolSchemaConverter
from llama_index.llms.google_genai.orchestration.chat_session import ChatSessionRunner
from llama_index.llms.google_genai.orchestration.structured import StructuredRunner


@pytest.fixture
def mock_genai_client() -> google.genai.Client:
    """Provides a strict mock of the google.genai.Client."""
    mock = create_autospec(google.genai.Client, instance=True)

    # Mock nested attributes which create_autospec might miss if they are dynamic
    # or strictly defined in __init__ but we want easier access.
    mock.aio = MagicMock()
    mock.aio.models = AsyncMock()
    mock.aio.chats = AsyncMock()
    mock.aio.files = AsyncMock()

    mock.models = MagicMock()
    mock.chats = MagicMock()
    mock.files = MagicMock()

    return mock


@pytest.fixture
def mock_file_manager() -> FileManager:
    return create_autospec(FileManager, instance=True)


@pytest.fixture
def mock_message_converter() -> MessageConverter:
    return create_autospec(MessageConverter, instance=True)


@pytest.fixture
def mock_response_converter() -> ResponseConverter:
    return create_autospec(ResponseConverter, instance=True)


@pytest.fixture
def mock_tool_schema_converter() -> ToolSchemaConverter:
    return create_autospec(ToolSchemaConverter, instance=True)


@pytest.fixture
def mock_chat_runner() -> ChatSessionRunner:
    return create_autospec(ChatSessionRunner, instance=True)


@pytest.fixture
def mock_structured_runner() -> StructuredRunner:
    return create_autospec(StructuredRunner, instance=True)
