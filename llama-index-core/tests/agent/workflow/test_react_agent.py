import pytest
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
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


@pytest.mark.asyncio
async def test_react_agent_keeps_a_system_message_from_chat_history():
    """
    A system message passed through chat_history reaches the LLM.

    The agent prepends its own system_prompt to the LLM input and the formatter
    renders it into the ReAct header, so that one is dropped as a duplicate.
    A caller-supplied system message is not that prompt and must not be dropped
    with it.
    """
    seen_messages = []

    def generator(messages, **kwargs):
        seen_messages.append(list(messages))
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Thought: done\nAnswer: ok",
        )

    agent = ReActAgent(
        llm=MockFunctionCallingLLM(response_generator=generator),
        tools=[],
    )
    workflow = AgentWorkflow(agents=[agent])

    await workflow.run(
        user_msg="hi",
        chat_history=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="Always answer in French.",
            ),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ],
    )

    assert seen_messages
    system_contents = [
        message.content or ""
        for message in seen_messages[0]
        if message.role == MessageRole.SYSTEM
    ]
    assert any("Always answer in French." in content for content in system_contents)
