import os
import pytest
import asyncio
from pydantic import BaseModel, Field
from llama_index.llms.genai import Gemini

import asyncio
import os

from llama_index.core.base.llms.types import (
    ChatResponse,
    ChatMessage,
    TextBlock,
    ImageBlock,
    MessageRole,
    CompletionResponse,
)
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_complete_and_acomplete() -> None:
    """Test both sync and async complete methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = "Write a poem about a magic backpack"

    # Test synchronous complete
    sync_response = llm.complete(prompt)
    assert sync_response is not None
    assert len(sync_response.text) > 0

    # Test async complete
    async_response = asyncio.run(llm.acomplete(prompt))
    assert async_response is not None
    assert len(async_response.text) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_chat_and_achat() -> None:
    """Test both sync and async chat methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous chat
    sync_response = llm.chat(messages=[message])
    assert sync_response is not None
    assert len(sync_response.message.content) > 0

    # Test async chat
    async_response = asyncio.run(llm.achat(messages=[message]))
    assert async_response is not None
    assert len(async_response.message.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_stream_chat_and_astream_chat() -> None:
    """Test both sync and async stream chat methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous stream chat
    sync_chunks = list(llm.stream_chat(messages=[message]))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in sync_chunks)

    # Test async stream chat
    async def test_async_stream() -> list[ChatResponse]:
        chunks = []
        async for chunk in await llm.astream_chat(messages=[message]):
            chunks.append(chunk)
        return chunks

    async_chunks = asyncio.run(test_async_stream())
    assert len(async_chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in async_chunks)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_stream_complete_and_astream_complete() -> None:
    """Test both sync and async stream complete methods."""
    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = "Write a poem about a magic backpack"

    # Test synchronous stream complete
    sync_chunks = list(llm.stream_complete(prompt))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in sync_chunks)

    # Test async stream complete
    async def test_async_stream() -> list[CompletionResponse]:
        chunks = []
        async for chunk in await llm.astream_complete(prompt):
            chunks.append(chunk)
        return chunks

    async_chunks = asyncio.run(test_async_stream())
    assert len(async_chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in async_chunks)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_simple_structured_predict() -> None:
    """Test structured prediction with a simple schema."""

    class Poem(BaseModel):
        content: str

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = llm.structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_complex_structured_predict() -> None:
    """Test structured prediction with a complex nested schema."""

    class Column(BaseModel):
        name: str = Field(description="Column field")
        data_type: str = Field(description="Data type field")

    class Table(BaseModel):
        name: str = Field(description="Table name field")
        columns: list[Column] = Field(description="List of random Column objects")

    class Schema(BaseModel):
        schema_name: str = Field(description="Schema name")
        tables: list[Table] = Field(description="List of random Table objects")

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = PromptTemplate("Generate a simple database structure")
    response = llm.structured_predict(output_cls=Schema, prompt=prompt)

    assert response is not None
    assert isinstance(response, Schema)
    assert isinstance(response.schema_name, str)
    assert len(response.schema_name) > 0
    assert len(response.tables) > 0
    assert all(isinstance(table, Table) for table in response.tables)
    assert all(len(table.columns) > 0 for table in response.tables)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_as_structured_llm() -> None:
    class Poem(BaseModel):
        content: str

    class Column(BaseModel):
        name: str = Field(description="Column field")
        data_type: str = Field(description="Data type field")

    class Table(BaseModel):
        name: str = Field(description="Table name field")
        columns: list[Column] = Field(description="List of random Column objects")

    class Schema(BaseModel):
        schema_name: str = Field(description="Schema name")
        tables: list[Table] = Field(description="List of random Table objects")

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = PromptTemplate("Generate content")

    # Test with simple schema
    poem_response = llm.as_structured_llm(output_cls=Poem, prompt=prompt).complete(
        "Write a poem about a magic backpack"
    )
    assert isinstance(poem_response.raw, Poem)
    assert len(poem_response.raw.content) > 0

    # Test with complex schema
    schema_response = llm.as_structured_llm(output_cls=Schema, prompt=prompt).complete(
        "Generate a simple database structure"
    )
    assert isinstance(schema_response.raw, Schema)
    assert len(schema_response.raw.schema_name) > 0
    assert len(schema_response.raw.tables) > 0


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_structured_predict_multiple_block() -> None:
    chat_messaages = [
        ChatMessage(
            content=[
                TextBlock(text="which logo is this?"),
                ImageBlock(
                    url="https://upload.wikimedia.org/wikipedia/commons/7/7a/Nohat-wiki-logo.png"
                ),
            ],
            role=MessageRole.USER,
        ),
    ]

    class Response(BaseModel):
        answer: str

    llm = Gemini(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    support = llm.structured_predict(
        output_cls=Response, prompt=ChatPromptTemplate(message_templates=chat_messaages)
    )
    assert isinstance(support, Response)
    assert "wiki" in support.answer.lower()
