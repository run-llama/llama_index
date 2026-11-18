import pytest

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.core.memory import Memory, StaticMemoryBlock
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
)
from llama_index.core.prompts import PromptTemplate


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


CALLS: list = []
FINAL_ANSWER = "Thought: I can answer now\nAnswer: The delivery charge is 7 USD."


class RecordingLLM(MockLLM):
    """Mock LLM that records the messages it receives and returns a parseable answer."""

    def chat(self, messages, **kwargs):
        CALLS.append(list(messages))
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, blocks=[TextBlock(text=FINAL_ANSWER)]
            )
        )


@pytest.mark.asyncio
async def test_react_agent_keeps_memory_block_content():
    """The stripped system message can carry memory-block content; it must reach the model."""
    CALLS.clear()
    memory = Memory.from_defaults(
        memory_blocks=[
            StaticMemoryBlock(
                name="delivery_policy",
                static_content="Our delivery charge is 7 USD per order.",
                priority=0,
            )
        ]
    )
    agent = ReActAgent(llm=RecordingLLM(), tools=[], streaming=False)

    await agent.run(user_msg="How much is delivery?", memory=memory)

    assert CALLS, "the model should have been called"
    first_input = "\n".join(str(m.content) for m in CALLS[0])
    assert "Our delivery charge is 7 USD per order." in first_input
