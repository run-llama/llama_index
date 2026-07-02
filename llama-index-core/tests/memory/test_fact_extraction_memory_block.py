from typing import Any, List, Sequence

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory.memory_blocks.fact import (
    DEFAULT_FACT_CONDENSE_PROMPT,
    FactExtractionMemoryBlock,
)


class _ScriptedLLM(MockLLM):
    """A MockLLM that returns pre-scripted responses, one per ``achat`` call."""

    scripted_responses: List[str] = Field(default_factory=list)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        content = self.scripted_responses.pop(0)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content)
        )


def test_condense_prompt_requests_full_deduplicated_snapshot() -> None:
    """The condense prompt must ask for the complete deduplicated snapshot.

    The condensed output replaces the stored facts wholesale, so the prompt has
    to request the full list - not just newly added facts - and must not carry
    the ambiguous 'do not duplicate ... existing facts' instruction that implied
    incremental (delta-only) output. See issue #21103.
    """
    messages = DEFAULT_FACT_CONDENSE_PROMPT.format_messages(
        existing_facts="<fact>user likes tea</fact>", max_facts=5
    )
    rendered = "\n".join(str(message.content) for message in messages)

    # The existing facts are injected into the prompt.
    assert "user likes tea" in rendered
    # It positively states the output replaces the stored facts (full snapshot).
    assert "completely replaces the existing facts" in rendered
    # The ambiguous, incremental-implying instructions are gone.
    assert (
        "Do not duplicate facts that are already in the existing facts list"
        not in rendered
    )
    assert "If no new facts are present" not in rendered


def test_condense_branch_replaces_facts_wholesale() -> None:
    """Plumbing-level contract the condense prompt is written to satisfy: once the
    fact list exceeds max_facts, the condensed response REPLACES the stored facts
    rather than being appended. (A real LLM is required to verify the prompt
    itself yields a full snapshot; this only pins the downstream behavior.)
    """
    llm = _ScriptedLLM(
        scripted_responses=[
            "<facts><fact>d</fact></facts>",  # extraction adds one new fact
            "<facts><fact>x</fact><fact>y</fact></facts>",  # condensed snapshot
        ]
    )
    block = FactExtractionMemoryBlock(llm=llm, facts=["a", "b", "c"], max_facts=3)

    # Putting a message triggers extraction (-> 4 facts > max 3) then condense.
    asyncio_run(block._aput([ChatMessage(role=MessageRole.USER, content="hello")]))

    # The condensed snapshot fully replaces the previous facts.
    assert block.facts == ["x", "y"]


def test_condense_empty_response_preserves_existing_facts() -> None:
    """An empty or unparseable condense response must not wipe the stored facts."""
    llm = _ScriptedLLM(
        scripted_responses=[
            "<facts><fact>d</fact></facts>",  # extraction pushes list over max
            "<facts></facts>",  # empty condense response
        ]
    )
    block = FactExtractionMemoryBlock(llm=llm, facts=["a", "b", "c"], max_facts=3)

    asyncio_run(block._aput([ChatMessage(role=MessageRole.USER, content="hello")]))

    # The empty condense output is ignored; the pre-condense facts are kept.
    assert block.facts == ["a", "b", "c", "d"]
