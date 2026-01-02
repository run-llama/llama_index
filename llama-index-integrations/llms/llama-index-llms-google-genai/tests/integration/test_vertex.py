import os
import pytest
from typing import List, Optional, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from unittest.mock import MagicMock

from llama_index.llms.google_genai import GoogleGenAI
import google.genai.types as types

# Conditions for running live tests
SKIP_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false") == "false"


@pytest.mark.skipif(SKIP_VERTEXAI, reason="GOOGLE_GENAI_USE_VERTEXAI not set")
def test_anyof_supported_vertexai() -> None:
    class Content(BaseModel):
        content: Union[int, str]

    llm = GoogleGenAI(model="gemini-2.5-flash-lite")

    content = (
        llm.as_structured_llm(output_cls=Content)
        .complete(prompt="Generate a small content")
        .raw
    )
    assert isinstance(content, Content)
    assert isinstance(content.content, int | str)


@pytest.mark.skipif(SKIP_VERTEXAI, reason="GOOGLE_GENAI_USE_VERTEXAI not set")
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

    llm = GoogleGenAI(model="gemini-2.5-flash-lite")

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


@pytest.mark.skipif(SKIP_VERTEXAI, reason="GOOGLE_GENAI_USE_VERTEXAI not set")
def test_cached_content_initialization_vertexai() -> None:
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite", cached_content=cached_content_value
    )

    assert llm.cached_content == cached_content_value
    assert llm._generation_config["cached_content"] == cached_content_value


def test_cached_content_in_response_vertexai() -> None:
    """
    This is pure response conversion behavior and doesn't require Vertex credentials.
    """
    from llama_index.llms.google_genai.conversion.responses import (
        ResponseConverter,
        GeminiResponseParseState,
    )

    converter = ResponseConverter()
    state = GeminiResponseParseState()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"
    mock_candidate.content.parts = [types.Part(text="Test response")]
    mock_response.candidates = [mock_candidate]
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.cached_content = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    chat_response = converter.to_chat_response(mock_response, state=state)
    assert "cached_content" in chat_response.raw


def test_cached_content_without_cached_content_vertexai() -> None:
    from llama_index.llms.google_genai.conversion.responses import (
        ResponseConverter,
        GeminiResponseParseState,
    )

    converter = ResponseConverter()
    state = GeminiResponseParseState()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"
    mock_candidate.content.parts = [types.Part(text="Test response")]
    mock_response.candidates = [mock_candidate]
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    # no cached_content
    if hasattr(mock_response, "cached_content"):
        delattr(mock_response, "cached_content")

    chat_response = converter.to_chat_response(mock_response, state=state)
    assert "cached_content" not in chat_response.raw


@pytest.mark.skipif(SKIP_VERTEXAI, reason="GOOGLE_GENAI_USE_VERTEXAI not set")
def test_cached_content_with_generation_config_vertexai() -> None:
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-456"

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        cached_content=cached_content_value,
        generation_config=types.GenerateContentConfig(
            temperature=0.5,
            cached_content=cached_content_value,
        ),
    )

    assert llm._generation_config["cached_content"] == cached_content_value
    assert llm._generation_config["temperature"] == 0.5
