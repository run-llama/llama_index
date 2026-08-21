import pytest

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
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


def _no_thought_agent(content: str) -> ReActAgent:
    return ReActAgent(
        name="calculator",
        description="Answers questions",
        tools=[],
        llm=MockFunctionCallingLLM(
            response_generator=lambda messages, **kwargs: ChatMessage(
                role=MessageRole.ASSISTANT, content=content
            )
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "I worked it out.\nAnswer: The sum is 8",
        "Answer: The sum is 8",
    ],
    ids=["preamble_instead_of_thought", "answer_only"],
)
async def test_react_agent_answer_without_thought(content: str) -> None:
    # The LLM always replies in the same shape, so a failed parse loops to max iterations.
    agent = _no_thought_agent(content)

    response = await agent.run(user_msg="Can you add 5 and 3?")

    assert "The sum is 8" in str(response.response)
