import pytest
from typing import List

from llama_index.core.memory.memory_blocks.fact import (
    FactExtractionMemoryBlock,
    DEFAULT_FACT_EXTRACT_PROMPT,
    DEFAULT_FACT_CONDENSE_PROMPT,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import MockLLM


class MyMockLLM(MockLLM):
    """Test-specific subclass of MockLLM with mocked achat method."""

    def __init__(self, *args, responses: List[ChatResponse], **kwargs):
        super().__init__(*args, **kwargs)
        self._responses = responses
        self._index = 0

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        response = self._responses[self._index]
        self._index += 1
        return response


@pytest.fixture
def mock_extraction_llm():
    """Create a mock LLM with extraction responses."""
    return MyMockLLM(
        responses=[
            ChatResponse(
                message=ChatMessage(
                    content="<facts><fact>John lives in New York</fact><fact>John is a software engineer</fact></facts>"
                )
            ),
        ]
    )


@pytest.fixture
def sample_messages():
    """Create sample chat messages."""
    return [
        ChatMessage(
            role=MessageRole.USER, content="My name is John and I live in New York."
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="Nice to meet you John!"),
        ChatMessage(
            role=MessageRole.USER,
            content="I work as a software engineer and I'm allergic to peanuts.",
        ),
    ]


@pytest.mark.asyncio
async def test_initialization():
    """Test initialization of FactExtractionMemoryBlock."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())
    assert memory_block.facts == []
    assert memory_block.fact_extraction_prompt_template == DEFAULT_FACT_EXTRACT_PROMPT
    assert memory_block.fact_condense_prompt_template == DEFAULT_FACT_CONDENSE_PROMPT

    # Test with custom prompt
    custom_prompt = "Custom prompt"
    memory_block = FactExtractionMemoryBlock(
        fact_extraction_prompt_template=custom_prompt
    )
    assert memory_block.fact_extraction_prompt_template.template == custom_prompt


@pytest.mark.asyncio
async def test_aget_empty_facts():
    """Test aget method when no facts are present."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())
    result = await memory_block.aget()
    assert result == ""


@pytest.mark.asyncio
async def test_aget_with_facts():
    """Test aget method with existing facts."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())
    memory_block.facts = ["John lives in New York", "John is a software engineer"]

    result = await memory_block.aget()
    assert (
        result
        == "<fact>John lives in New York</fact>\n<fact>John is a software engineer</fact>"
    )


@pytest.mark.asyncio
async def test_aput_with_mocked_response(mock_extraction_llm, sample_messages):
    """Test aput method with a mocked LLM response."""
    memory_block = FactExtractionMemoryBlock(llm=mock_extraction_llm)

    await memory_block.aput(sample_messages)

    # Verify the extracted facts
    assert len(memory_block.facts) == 2
    assert "John lives in New York" in memory_block.facts
    assert "John is a software engineer" in memory_block.facts


@pytest.mark.asyncio
async def test_aput_with_empty_messages():
    """Test aput method with empty messages."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())

    await memory_block.aput([])
    assert memory_block.facts == []


@pytest.mark.asyncio
async def test_aput_with_duplicate_facts(mock_extraction_llm, sample_messages):
    """Test aput method with duplicate facts."""
    memory_block = FactExtractionMemoryBlock(llm=mock_extraction_llm)
    memory_block.facts = ["John lives in New York"]

    await memory_block.aput(sample_messages)

    # Verify only new facts were added (no duplicates)
    assert len(memory_block.facts) == 2
    assert memory_block.facts == [
        "John lives in New York",
        "John is a software engineer",
    ]


@pytest.mark.asyncio
async def test_parse_facts_xml():
    """Test the _parse_facts_xml method."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())

    xml_text = """
    <facts>
      <fact>John lives in New York</fact>
      <fact>John is a software engineer</fact>
      <fact>John is allergic to peanuts</fact>
    </facts>
    """

    facts = memory_block._parse_facts_xml(xml_text)

    assert len(facts) == 3
    assert facts[0] == "John lives in New York"
    assert facts[1] == "John is a software engineer"
    assert facts[2] == "John is allergic to peanuts"


@pytest.mark.asyncio
async def test_parse_facts_xml_with_empty_response():
    """Test the _parse_facts_xml method with an empty response."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())

    xml_text = "<facts></facts>"

    facts = memory_block._parse_facts_xml(xml_text)

    assert facts == []


@pytest.mark.asyncio
async def test_parse_facts_xml_with_malformed_xml():
    """Test the _parse_facts_xml method with malformed XML."""
    memory_block = FactExtractionMemoryBlock(llm=MockLLM())

    xml_text = """
    Some text without proper XML tags
    <fact>This should be extracted</fact>
    More text
    <fact>This should also be extracted</fact>
    """

    facts = memory_block._parse_facts_xml(xml_text)

    assert len(facts) == 2
    assert facts[0] == "This should be extracted"
    assert facts[1] == "This should also be extracted"
