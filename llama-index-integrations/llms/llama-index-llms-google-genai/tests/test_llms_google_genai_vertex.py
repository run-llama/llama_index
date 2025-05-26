from datetime import datetime
from enum import Enum
import os
from typing import List, Optional, Union

import pytest
from llama_index.core.program.function_program import get_function_tool
from pydantic import BaseModel, Field

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai.utils import convert_schema_to_function_declaration

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
