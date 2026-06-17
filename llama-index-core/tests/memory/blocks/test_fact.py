import pytest
from typing import Any, List

from llama_index.core.memory.memory_blocks.fact import (
    FactExtractionMemoryBlock,
    DEFAULT_FACT_EXTRACT_PROMPT,
    DEFAULT_FACT_CONDENSE_PROMPT,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import MockLLM


class MyMockLLM(MockLLM):
    """Test-specific subclass of MockLLM with mocked achat method."""

    def __init__(self, *args, responses: List[ChatResponse], **kwargs):
        super().__init__(*args, **kwargs)
        self._responses = responses
        self._index = 0
        self._messages_history: List[List[ChatMessage]] = []

    @property
    def messages_history(self) -> List[List[ChatMessage]]:
        return self._messages_history

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        self.messages_history.append(messages)
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


@pytest.fixture
def tool_history_messages():
    """Create messages with provider-native tool content."""
    return [
        ChatMessage(role=MessageRole.USER, content="Find my next flight."),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="call_123",
                    tool_name="lookup_flight",
                    tool_kwargs={"destination": "Paris"},
                )
            ],
            additional_kwargs={"tool_calls": [{"name": "lookup_flight"}]},
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content="Flight AF123 leaves at 5pm.",
            additional_kwargs={"tool_call_id": "call_123"},
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
async def test_aput_flattens_tool_history_before_fact_extraction(
    tool_history_messages,
):
    """Test aput flattens tool blocks before calling the fact LLM."""
    mock_llm = MyMockLLM(
        responses=[
            ChatResponse(
                message=ChatMessage(
                    content="<facts><fact>User asked about a flight</fact></facts>"
                )
            ),
        ]
    )
    memory_block = FactExtractionMemoryBlock(llm=mock_llm)

    await memory_block.aput(tool_history_messages)

    assert len(mock_llm.messages_history) == 1
    fact_messages = mock_llm.messages_history[0]
    assert all(message.role != MessageRole.TOOL for message in fact_messages)
    assert all(
        not isinstance(block, ToolCallBlock)
        for message in fact_messages
        for block in message.blocks
    )

    conversation_message = fact_messages[0]
    assert conversation_message.role == MessageRole.USER
    assert len(conversation_message.blocks) == 1
    assert isinstance(conversation_message.blocks[0], TextBlock)
    conversation_text = conversation_message.blocks[0].text
    assert "<message role='user'>Find my next flight.</message>" in conversation_text
    assert "<message role='tool'>Flight AF123 leaves at 5pm." in conversation_text
    assert "lookup_flight" not in conversation_text
    assert "tool_call_id" not in conversation_text
    assert memory_block.facts == ["User asked about a flight"]


@pytest.mark.asyncio
async def test_aput_flattens_tool_history_before_condensing(
    tool_history_messages,
):
    """Test condensing uses the same flattened history as extraction."""
    mock_llm = MyMockLLM(
        responses=[
            ChatResponse(
                message=ChatMessage(
                    content=(
                        "<facts>"
                        "<fact>User asked about a flight</fact>"
                        "<fact>Flight AF123 leaves at 5pm</fact>"
                        "</facts>"
                    )
                )
            ),
            ChatResponse(
                message=ChatMessage(
                    content="<facts><fact>User asked about AF123</fact></facts>"
                )
            ),
        ]
    )
    memory_block = FactExtractionMemoryBlock(llm=mock_llm, max_facts=1)

    await memory_block.aput(tool_history_messages)

    assert len(mock_llm.messages_history) == 2
    for llm_messages in mock_llm.messages_history:
        assert all(message.role != MessageRole.TOOL for message in llm_messages)
        assert all(
            not isinstance(block, ToolCallBlock)
            for message in llm_messages
            for block in message.blocks
        )
        assert llm_messages[0].role == MessageRole.USER
        conversation_text = llm_messages[0].content or ""
        assert "Flight AF123 leaves at 5pm." in conversation_text
        assert "lookup_flight" not in conversation_text
        assert "tool_call_id" not in conversation_text

    assert memory_block.facts == ["User asked about AF123"]


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
