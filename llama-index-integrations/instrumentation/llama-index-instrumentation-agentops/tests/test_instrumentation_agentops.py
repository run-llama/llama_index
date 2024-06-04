import pytest
import uuid
import llama_index.core.instrumentation as instrument
from agentops import LLMEvent
from typing import Any, Generator, Sequence
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.chatml_utils import completion_to_prompt, messages_to_prompt
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import LLM
from llama_index.instrumentation.agentops import AgentOpsHandler
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.bridge.pydantic import Field
from unittest.mock import patch, MagicMock


MOCK_AGENT_PROMPT = "This is a test prompt"
MOCK_AGENT_RESPONSE = "This is a test response"


dispatcher = instrument.get_dispatcher("test")


class MockLLM(CustomLLM):
    model: str = Field(default="MockLLMModelName")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(content=MOCK_AGENT_RESPONSE))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        ...


class MockAgentWorker(BaseAgentWorker):
    def __init__(self, llm: LLM):
        """Initialize."""
        self._llm = llm

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        counter = 0
        task.extra_state["counter"] = counter
        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            memory=task.memory,
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        chat_res = self._llm.chat([ChatMessage(content=task.input)])
        chat_agent_res = AgentChatResponse(chat_res.message.content)
        return TaskStepOutput(
            output=chat_agent_res,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        return self.run_step(step=step, task=task, **kwargs)

    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        chat_res = self._llm.chat([ChatMessage(content=task.input)])
        chat_agent_res = StreamingAgentChatResponse(chat_res.message.content)
        return TaskStepOutput(
            output=chat_agent_res,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        return self.stream_step(step=step, task=task, **kwargs)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""


def test_class():
    pass
    # TODO: Test for all final class names
    # names_of_base_classes = [b.__name__ for b in AgentOpsEventHandler.__mro__]
    # assert BaseEventHandler.__name__ in names_of_base_classes


@pytest.mark.asyncio()
@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat", "astream_chat"])
@patch("llama_index.instrumentation.agentops.base.AOClient")
async def test_agentops_event_handler_emits_llmevents(
    mock_ao_client: MagicMock, method: str
):
    mock_ao_client_instance = MagicMock()
    mock_ao_client.return_value = mock_ao_client_instance

    AgentOpsHandler.add_handler()
    mock_llm = MockLLM()
    agent_runner = AgentRunner(agent_worker=MockAgentWorker(llm=mock_llm))

    # Initiate a chat with the agent
    if method == "chat":
        agent_runner.chat(MOCK_AGENT_PROMPT)
    elif method == "achat":
        await agent_runner.achat(MOCK_AGENT_PROMPT)
    elif method == "stream_chat":
        agent_runner.stream_chat(MOCK_AGENT_PROMPT)
    else:
        await agent_runner.astream_chat(MOCK_AGENT_PROMPT)

    calls_to_agentops = mock_ao_client_instance.record.call_args_list
    assert len(calls_to_agentops) == 2
    agent_start_chat_event: LLMEvent = calls_to_agentops[0][0][0]
    agent_end_chat_event: LLMEvent = calls_to_agentops[1][0][0]

    # We expect one LLMEvent for the prompt, then one for the LLM response
    expected_agent_prompt = messages_to_prompt([ChatMessage(content=MOCK_AGENT_PROMPT)])
    expected_agent_completion = completion_to_prompt(MOCK_AGENT_RESPONSE)
    assert agent_start_chat_event.prompt == expected_agent_prompt
    assert not agent_start_chat_event.completion
    assert agent_start_chat_event.model == mock_llm.model
    assert agent_end_chat_event.prompt == expected_agent_prompt
    assert agent_end_chat_event.completion == expected_agent_completion
    assert agent_end_chat_event.model == mock_llm.model
