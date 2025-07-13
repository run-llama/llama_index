from datetime import datetime
from enum import Enum
import os
from typing import List, Optional, Union
from unittest.mock import MagicMock

import pytest
from llama_index.core.program.function_program import get_function_tool
from pydantic import BaseModel, Field

from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai.utils import (
    convert_schema_to_function_declaration,
    chat_from_gemini_response,
)

# Don't forget to export GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_PROJECT when testing with VertexAI
SKIP_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false") == "false"


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_anyof_supported_vertexai() -> None:
    class Content(BaseModel):
        content: Union[int, str]

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
    )
    function_tool = get_function_tool(Content)
    _ = convert_schema_to_function_declaration(llm._client, function_tool)

    content = (
        llm.as_structured_llm(output_cls=Content)
        .complete(prompt="Generate a small content")
        .raw
    )
    assert isinstance(content, Content)
    assert isinstance(content.content, int | str)


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_optional_lists_nested_vertexai() -> None:
    class Address(BaseModel):
        street: str
        city: str
        country: str = Field(default="USA")

    class ContactInfo(BaseModel):
        email: str
        phone: Optional[str] = None
        address: Address

    class Department(Enum):
        ENGINEERING = "engineering"
        MARKETING = "marketing"
        SALES = "sales"
        HR = "human_resources"

    class Employee(BaseModel):
        name: str
        contact: ContactInfo
        department: Department
        hire_date: datetime

    class Company(BaseModel):
        name: str
        founded_year: int
        website: str
        employees: List[Employee]
        headquarters: Address

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
    )

    function_tool = get_function_tool(Company)
    converted = convert_schema_to_function_declaration(llm._client, function_tool)

    assert converted.name == "Company"
    assert converted.description is not None
    assert converted.parameters.required is not None

    assert list(converted.parameters.properties) == [
        "name",
        "founded_year",
        "website",
        "employees",
        "headquarters",
    ]

    assert "name" in converted.parameters.required
    assert "founded_year" in converted.parameters.required
    assert "website" in converted.parameters.required
    assert "employees" in converted.parameters.required
    assert "headquarters" in converted.parameters.required

    # call the model and check the output
    company = (
        llm.as_structured_llm(output_cls=Company)
        .complete(prompt="Create a fake company with at least 3 employees")
        .raw
    )
    assert isinstance(company, Company)

    assert len(company.employees) >= 3
    assert all(
        employee.department in Department.__members__.values()
        for employee in company.employees
    )


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_cached_content_initialization_vertexai() -> None:
    """Test GoogleGenAI initialization with cached_content parameter in VertexAI."""
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    llm = GoogleGenAI(model="gemini-2.0-flash-001", cached_content=cached_content_value)

    # Verify cached_content is stored in the instance
    assert llm.cached_content == cached_content_value

    # Verify cached_content is stored in generation config
    assert llm._generation_config["cached_content"] == cached_content_value


def test_cached_content_in_response_vertexai() -> None:
    """Test that cached_content is extracted from Gemini responses in VertexAI."""
    # Mock response with cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Test response"
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    mock_response.cached_content = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify cached_content is in raw response
    assert "cached_content" in chat_response.raw
    assert (
        chat_response.raw["cached_content"]
        == "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"
    )


def test_cached_content_without_cached_content_vertexai() -> None:
    """Test response processing when cached_content is not present in VertexAI."""
    # Mock response without cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Test response"
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    # No cached_content attribute
    del mock_response.cached_content

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify no cached_content key in raw response
    assert "cached_content" not in chat_response.raw


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_cached_content_with_generation_config_vertexai() -> None:
    """Test that cached_content works with custom generation_config in VertexAI."""
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-456"

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        cached_content=cached_content_value,
        generation_config=types.GenerateContentConfig(
            temperature=0.5,
            cached_content=cached_content_value,
        ),
    )

    # Verify both cached_content and custom config are preserved
    assert llm._generation_config["cached_content"] == cached_content_value
    assert llm._generation_config["temperature"] == 0.5


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_google_search_basic_vertexai() -> None:
    """Test basic Google Search functionality with VertexAI."""
    llm = GoogleGenAI(model="gemini-2.0-flash-001")
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    response = llm.chat_with_tools(
        user_msg="What is the latest news about artificial intelligence?",
        tools=[grounding_tool],
    )

    assert response is not None
    assert response.message.content is not None
    assert len(response.message.content) > 0


@pytest.mark.skipif(
    SKIP_VERTEXAI,
    reason="GOOGLE_GENAI_USE_VERTEXAI not set",
)
def test_google_search_grounding_metadata_vertexai() -> None:
    """Test that Google Search returns comprehensive grounding metadata in VertexAI response."""
    llm = GoogleGenAI(model="gemini-2.0-flash-001")
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    response = llm.chat_with_tools(
        user_msg="What is the capital of Japan?",
        tools=[grounding_tool],
    )

    assert response is not None
    assert response.message.content is not None
    assert len(response.message.content) > 0

    raw_response = response.raw
    assert raw_response is not None
    assert isinstance(raw_response, dict)

    # Grounding metadata must always be present
    assert "grounding_metadata" in raw_response
    assert raw_response["grounding_metadata"] is not None
    grounding_metadata = raw_response["grounding_metadata"]
    assert isinstance(grounding_metadata, dict)

    # Web search queries must always be present
    assert "web_search_queries" in grounding_metadata
    assert grounding_metadata["web_search_queries"] is not None
    assert isinstance(grounding_metadata["web_search_queries"], list)
    assert len(grounding_metadata["web_search_queries"]) > 0

    # Validate each web search query
    for query in grounding_metadata["web_search_queries"]:
        assert isinstance(query, str)
        assert len(query.strip()) > 0

    # Search entry point must always be present
    assert "search_entry_point" in grounding_metadata
    assert grounding_metadata["search_entry_point"] is not None
    search_entry_point = grounding_metadata["search_entry_point"]
    assert isinstance(search_entry_point, dict)

    # Rendered content must always be present
    assert "rendered_content" in search_entry_point
    assert search_entry_point["rendered_content"] is not None
    assert isinstance(search_entry_point["rendered_content"], str)
    assert len(search_entry_point["rendered_content"].strip()) > 0

    # Grounding supports must always be present
    assert "grounding_supports" in grounding_metadata
    assert grounding_metadata["grounding_supports"] is not None
    assert isinstance(grounding_metadata["grounding_supports"], list)
    assert len(grounding_metadata["grounding_supports"]) > 0

    # Validate each grounding support entry has required structure
    for support in grounding_metadata["grounding_supports"]:
        assert isinstance(support, dict)

        # Required fields that must be present
        assert "segment" in support
        assert "grounding_chunk_indices" in support
        segment = support["segment"]
        assert isinstance(segment, dict)

        # Segment fields must be present
        assert "start_index" in segment
        assert "end_index" in segment
        assert "part_index" in segment
        assert "text" in segment

    # Grounding chunks must always be present
    assert "grounding_chunks" in grounding_metadata
    assert grounding_metadata["grounding_chunks"] is not None
    assert isinstance(grounding_metadata["grounding_chunks"], list)
    assert len(grounding_metadata["grounding_chunks"]) > 0

    # Validate each grounding chunk has required structure
    for chunk in grounding_metadata["grounding_chunks"]:
        assert isinstance(chunk, dict)

        # Required fields that must be present
        assert "web" in chunk
        web_chunk = chunk["web"]
        assert isinstance(web_chunk, dict)

        # Web chunk fields must be present
        assert "uri" in web_chunk
        assert "title" in web_chunk
