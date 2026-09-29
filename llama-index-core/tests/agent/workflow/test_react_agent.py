import pytest

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent.workflow.react_agent import _dump_raw
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.workflow import Context
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


class _Model(BaseModel):
    a: int


class _KeyErrorRaw(dict):
    """Mimics SDK responses (e.g. DashScope) whose `__getattr__` raises KeyError."""

    def __getattr__(self, attr):
        return self[attr]


def test_dump_raw_handles_keyerror_getattr():
    raw = _KeyErrorRaw(output="hi")
    assert _dump_raw(raw) is raw
    assert _dump_raw(None) is None
    assert _dump_raw(_Model(a=1)) == {"a": 1}


@pytest.mark.asyncio
async def test_react_agent_take_step_with_keyerror_raw():
    class _LLM(MockLLM):
        async def achat(self, messages, **kwargs):
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", content="Thought: done\nAnswer: 42"
                ),
                raw=_KeyErrorRaw(output="hi"),
            )

    agent = ReActAgent(tools=[], llm=_LLM(), streaming=False)
    ctx = Context(agent)
    output = await agent.take_step(ctx, [ChatMessage(role="user", content="q")], [], [])
    assert output.raw == {"output": "hi"}
