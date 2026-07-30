from typing import Any, List, Sequence

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
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


class _RejectingToolHistoryLLM(_ScriptedLLM):
    """Reject provider-specific tool history and capture sanitized prompts."""

    captured_messages: List[List[ChatMessage]] = Field(default_factory=list)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        for message in messages:
            if message.role in (MessageRole.FUNCTION, MessageRole.TOOL):
                raise ValueError("provider rejected raw tool-result history")
            if any(isinstance(block, ToolCallBlock) for block in message.blocks):
                raise ValueError("provider rejected raw tool-call history")

        self.captured_messages.append(list(messages))
        return await super().achat(messages, **kwargs)


def _tool_history() -> List[ChatMessage]:
    return [
        ChatMessage(
            role=MessageRole.USER,
            content="Remember that the deployment uses <pgvector> & HNSW.",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                TextBlock(text="I will inspect the deployment."),
                ToolCallBlock(
                    tool_name="inspect_deployment",
                    tool_kwargs={"service": "retrieval"},
                    tool_call_id="call-1",
                ),
            ],
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content="pgvector is enabled",
            additional_kwargs={"tool_call_id": "call-1"},
        ),
    ]


def test_fact_extraction_serializes_tool_history_as_provider_neutral_text() -> None:
    llm = _RejectingToolHistoryLLM()
    llm.scripted_responses = ["<facts><fact>deployment uses pgvector</fact></facts>"]
    block = FactExtractionMemoryBlock(llm=llm)

    asyncio_run(block._aput(_tool_history()))

    assert block.facts == ["deployment uses pgvector"]
    assert len(llm.captured_messages) == 1
    rendered = "\n".join(message.content or "" for message in llm.captured_messages[0])
    assert "<current_conversation>" in rendered
    assert (
        "<user>Remember that the deployment uses &lt;pgvector&gt; &amp; HNSW.</user>"
        in rendered
    )
    assert "<assistant>I will inspect the deployment.</assistant>" in rendered
    assert "<tool>pgvector is enabled</tool>" in rendered
    assert "inspect_deployment" not in rendered


def test_fact_condense_serializes_tool_history_as_provider_neutral_text() -> None:
    llm = _RejectingToolHistoryLLM()
    llm.scripted_responses = [
        "<facts><fact>d</fact></facts>",
        "<facts><fact>x</fact><fact>y</fact></facts>",
    ]
    block = FactExtractionMemoryBlock(llm=llm, facts=["a", "b", "c"], max_facts=3)

    asyncio_run(block._aput(_tool_history()))

    assert block.facts == ["x", "y"]
    assert len(llm.captured_messages) == 2
    for messages in llm.captured_messages:
        assert all(
            message.role not in (MessageRole.FUNCTION, MessageRole.TOOL)
            for message in messages
        )
        assert all(
            not any(isinstance(block, ToolCallBlock) for block in message.blocks)
            for message in messages
        )


def test_condense_prompt_requests_full_deduplicated_snapshot() -> None:
    """
    The condense prompt must ask for the complete deduplicated snapshot.

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
    """
    Plumbing-level contract the condense prompt is written to satisfy: once the
    fact list exceeds max_facts, the condensed response REPLACES the stored facts
    rather than being appended. (A real LLM is required to verify the prompt
    itself yields a full snapshot; this only pins the downstream behavior.)
    """
    # MockLLM's __init__ has a fixed signature, so assign the subclass field
    # after construction (as in tests/llms/test_callbacks.py's _SecretMockLLM).
    llm = _ScriptedLLM()
    llm.scripted_responses = [
        "<facts><fact>d</fact></facts>",  # extraction adds one new fact
        "<facts><fact>x</fact><fact>y</fact></facts>",  # condensed snapshot
    ]
    block = FactExtractionMemoryBlock(llm=llm, facts=["a", "b", "c"], max_facts=3)

    # Putting a message triggers extraction (-> 4 facts > max 3) then condense.
    asyncio_run(block._aput([ChatMessage(role=MessageRole.USER, content="hello")]))

    # The condensed snapshot fully replaces the previous facts.
    assert block.facts == ["x", "y"]


def test_condense_empty_response_preserves_existing_facts() -> None:
    """An empty or unparsable condense response must not wipe the stored facts."""
    llm = _ScriptedLLM()
    llm.scripted_responses = [
        "<facts><fact>d</fact></facts>",  # extraction pushes list over max
        "<facts></facts>",  # empty condense response
    ]
    block = FactExtractionMemoryBlock(llm=llm, facts=["a", "b", "c"], max_facts=3)

    asyncio_run(block._aput([ChatMessage(role=MessageRole.USER, content="hello")]))

    # The empty condense output is ignored; the pre-condense facts are kept.
    assert block.facts == ["a", "b", "c", "d"]
