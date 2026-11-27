from typing import Any, Sequence

import pytest

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool


def test_react_agent_prompts():
    llm = MockLLM()
    agent = ReActAgent(
        llm=llm,
        tools=[],
    )

    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert isinstance(prompts["react_header"], PromptTemplate)

    new_prompt = "New prompt"
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt in str(prompts["react_header"])

    new_prompt = PromptTemplate("New prompt 2")
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt == prompts["react_header"]


@pytest.mark.asyncio
async def test_react_agent_return_direct_preserves_answer_substring():
    """A return_direct tool's output is returned verbatim, even if it contains 'Answer:'."""
    tool_output = "Sources:\n  doc-1: pricing faq\nAnswer: 42 dollars per seat"

    def lookup(query: str) -> str:
        """Search the knowledge base."""
        return tool_output

    llm = MockFunctionCallingLLM(
        response_generator=lambda messages, **kwargs: ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "Thought: I should search.\n"
                "Action: lookup\n"
                'Action Input: {"query": "seat price"}'
            ),
        )
    )

    agent = ReActAgent(
        llm=llm,
        tools=[FunctionTool.from_defaults(fn=lookup, return_direct=True)],
        streaming=False,
    )

    result = await agent.run(user_msg="how much is a seat?")

    assert result.response.content == tool_output


class _FixedTextLLM(MockLLM):
    """Mock LLM whose achat always returns the class-level llm_text."""

    llm_text: str = ""

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.llm_text)
        )


class _HandoffToBLLM(_FixedTextLLM):
    """Always emits a handoff action to agent B."""

    llm_text: str = (
        "Thought: B should handle this.\n"
        "Action: handoff\n"
        'Action Input: {"to_agent": "B", "reason": "b knows"}'
    )


class _AnswerLLM(_FixedTextLLM):
    """Always emits a final answer."""

    llm_text: str = "Thought: I know this.\nAnswer: B final answer 42"


@pytest.mark.asyncio
async def test_react_agent_handoff_still_strips_answer_scaffolding():
    """After a handoff, the answering agent's own LLM text keeps the 'Answer:' cleanup."""
    workflow = AgentWorkflow(
        agents=[
            ReActAgent(
                name="A",
                description="agent a",
                llm=_HandoffToBLLM(),
                streaming=False,
            ),
            ReActAgent(
                name="B",
                description="agent b",
                llm=_AnswerLLM(),
                streaming=False,
            ),
        ],
        root_agent="A",
    )

    handler = workflow.run(user_msg="hello")
    async for _ in handler.stream_events():
        pass

    result = await handler

    assert result.response.content == "B final answer 42"
