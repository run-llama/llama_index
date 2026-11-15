import pytest

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import MockLLM
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


class RecordingMockLLM(MockLLM):
    """Mock LLM that records every message list it is called with."""

    seen: list = Field(default_factory=list)
    answer: str = "Thought: I can answer directly.\nAnswer: Hello there."

    def _record(self, messages) -> None:
        self.seen.append([(message.role.value, message.content) for message in messages])

    def _response(self) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self.answer)
        )

    def chat(self, messages, **kwargs) -> ChatResponse:
        self._record(messages)
        return self._response()

    async def achat(self, messages, **kwargs) -> ChatResponse:
        self._record(messages)
        return self._response()


@pytest.mark.asyncio
async def test_react_agent_keeps_system_message_from_chat_history():
    """A system message passed through chat history must reach the LLM."""
    llm = RecordingMockLLM()
    agent = ReActAgent(llm=llm, tools=[], streaming=False)
    workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

    await workflow.run(
        user_msg="Say hello.",
        chat_history=[ChatMessage(role="system", content="You are a pirate.")],
    )

    assert llm.seen, "The LLM was never called"
    system_messages = [
        content for role, content in llm.seen[0] if role == "system"
    ]
    assert len(system_messages) == 1
    assert "You are a pirate." in (system_messages[0] or "")
