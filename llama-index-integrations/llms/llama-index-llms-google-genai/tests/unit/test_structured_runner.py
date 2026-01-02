import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel
import google.genai.types as types
from llama_index.llms.google_genai.orchestration.structured import (
    StructuredRunner,
    StructuredStreamParser,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class MyModel(BaseModel):
    foo: str
    bar: int


@pytest.fixture
def structured_runner(mock_genai_client, mock_file_manager, mock_message_converter):
    return StructuredRunner(
        client=mock_genai_client,
        model="gemini-2.0-flash",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
    )


def test_parser_valid_json():
    parser = StructuredStreamParser(output_cls=MyModel)

    # Simulate a chunk with parsed object (native SDK behavior)
    chunk = MagicMock()
    chunk.parsed = MyModel(foo="hello", bar=123)
    chunk.candidates = []

    result = parser.update(chunk)
    assert result == chunk.parsed


def test_parser_text_accumulation():
    parser = StructuredStreamParser(output_cls=MyModel)

    # Chunk 1: "{"
    chunk1 = MagicMock()
    chunk1.parsed = None
    part1 = MagicMock()
    part1.text = '{"foo": "he'
    chunk1.candidates = [MagicMock(content=MagicMock(parts=[part1]))]

    # Chunk 2: "llo", "bar": 123}"
    chunk2 = MagicMock()
    chunk2.parsed = None
    part2 = MagicMock()
    part2.text = 'llo", "bar": 123}'
    chunk2.candidates = [MagicMock(content=MagicMock(parts=[part2]))]

    # First update shouldn't return anything valid yet (incomplete JSON)
    # Note: The FlexibleModel might try to repair it, but let's assume strict fail first
    res1 = parser.update(chunk1)

    # Second update completes it
    res2 = parser.update(chunk2)

    assert isinstance(res2, MyModel)
    assert res2.foo == "hello"
    assert res2.bar == 123


@pytest.mark.asyncio
async def test_prepare_config(structured_runner):
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    # Mock converter
    structured_runner._message_converter.to_gemini_content.return_value = (
        types.Content(parts=[types.Part(text="Hello")]),
        [],
    )

    prepared = await structured_runner.prepare(
        messages=messages, output_cls=MyModel, generation_config={"temperature": 0.5}
    )

    assert prepared.config.response_mime_type == "application/json"
    assert prepared.config.response_schema == MyModel
    assert prepared.config.temperature == 0.5


@pytest.mark.asyncio
async def test_arun_parsed_response(structured_runner, mock_genai_client):
    structured_runner._file_manager.file_mode = "hybrid"

    prepared = MagicMock()
    prepared.model = "gemini-2.0-flash"
    prepared.contents = [types.Content(parts=[types.Part(text="Hello")])]
    prepared.config = types.GenerateContentConfig(response_mime_type="application/json")
    prepared.uploaded_file_names = ["file-1"]
    prepared.output_cls = MyModel

    # Mock client response
    mock_response = MagicMock()
    mock_response.parsed = MyModel(foo="success", bar=1)
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    result = await structured_runner.arun(prepared)

    assert result.foo == "success"
    assert result.bar == 1

    # Verify cleanup called
    structured_runner._file_manager.acleanup.assert_called_once_with(["file-1"])
