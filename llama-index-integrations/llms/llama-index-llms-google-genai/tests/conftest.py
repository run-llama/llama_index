import os
from typing import List, Optional

import pytest
from unittest.mock import create_autospec, MagicMock, AsyncMock, patch
import google.genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from pydantic import BaseModel, Field

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai.files import FileManager
from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.conversion.responses import ResponseConverter
from llama_index.llms.google_genai.conversion.tools import ToolSchemaConverter
from llama_index.llms.google_genai.orchestration.chat_session import ChatSessionRunner
from llama_index.llms.google_genai.orchestration.structured import StructuredRunner


# Conditions for running live tests
SKIP_GEMINI = (
    os.environ.get("GOOGLE_API_KEY") is None
    or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false") == "true"
)


class Poem(BaseModel):
    content: str


class Column(BaseModel):
    name: str = Field(description="Column field")
    data_type: str = Field(description="Data type field")


class Table(BaseModel):
    name: str = Field(description="Table name field")
    columns: List[Column] = Field(description="List of random Column objects")


class Schema(BaseModel):
    schema_name: str = Field(description="Schema name")
    tables: List[Table] = Field(description="List of random Table objects")


class TextContent(BaseModel):
    """A piece of text content."""

    text: str
    language: str


class ImageContent(BaseModel):
    """A piece of image content."""

    url: str
    alt_text: Optional[str]
    width: Optional[int]
    height: Optional[int]


class VideoContent(BaseModel):
    """A piece of video content."""

    url: str
    duration_seconds: int
    thumbnail: Optional[str]


class Content(BaseModel):
    """Content of a blog post."""

    title: str
    created_at: str
    text: Optional[TextContent] = None
    image: Optional[ImageContent]
    video: Optional[VideoContent]
    tags: List[str]


class BlogPost(BaseModel):
    """A blog post."""

    id: str
    author: str
    published: bool
    contents: List[Content]
    category: Optional[str]


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


GEMINI_MODELS_TO_TEST = (
    [
        {"model": "gemini-2.5-flash-lite", "config": {}},
        {
            "model": "gemini-2.5-flash-lite",
            "config": {
                "generation_config": GenerateContentConfig(
                    thinking_config=ThinkingConfig(thinking_budget=512)
                )
            },
        },
    ]
    if not SKIP_GEMINI
    else []
)


@pytest.fixture(params=GEMINI_MODELS_TO_TEST)
def llm(request) -> GoogleGenAI:
    return GoogleGenAI(
        model=request.param["model"],
        api_key=os.environ["GOOGLE_API_KEY"],
        **request.param.get("config", {}),
    )


@pytest.fixture
def mock_genai_client_factory(mock_genai_client):
    """Patches the GenAIClientFactory to return our strict mock client."""
    with patch(
        "llama_index.llms.google_genai.client.GenAIClientFactory.create"
    ) as mock_create:
        # The factory returns (client, model_meta)
        mock_model_meta = MagicMock()
        mock_model_meta.output_token_limit = 8192
        mock_model_meta.input_token_limit = 200000

        mock_create.return_value = (mock_genai_client, mock_model_meta)
        yield mock_create


@pytest.fixture
def mocked_llm(mock_genai_client_factory):
    """Returns a GoogleGenAI instance using the mocked client."""
    return GoogleGenAI(model="gemini-2.5-flash-lite", api_key="dummy")
