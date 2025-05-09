from typing import Any

from unittest.mock import patch, MagicMock
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)

from llama_index.agent.introspective.reflective.tool_interactive_reflection import (
    Correction,
    ToolInteractiveReflectionAgentWorker,
)

PRINT_CHAT_HISTORY = False

mock_correction = Correction(correction="This is a mock correction.")


class MockLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """
        LLM metadata.

        Returns:
            LLMMetadata: LLM metadata containing various information about the LLM.

        """
        return LLMMetadata()

    def structured_predict(
        self, output_cls: BaseModel, prompt: PromptTemplate, **prompt_args: Any
    ) -> BaseModel:
        """This is fixed so that it goes through 2 Reflections and 1 Correction."""
        if output_cls == Correction:
            return mock_correction
        else:
            raise ValueError("Unexpected output_cls type for this test.")

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


@patch("llama_index.core.agent.function_calling.step.FunctionCallingAgentWorker")
def test_introspective_agent_with_stopping_callable(mock_critique_agent_worker) -> None:
    # Arrange
    mock_critique_agent = MagicMock()
    mock_critique_agent.chat.side_effect = [
        AgentChatResponse(response="This is a mock critique str."),
        AgentChatResponse(response="This is another mock critique str."),
    ]
    mock_stopping_callable = MagicMock()
    mock_stopping_callable.side_effect = [False, True]
    dummy_llm = MockLLM()

    mock_critique_agent_worker.as_agent.return_value = mock_critique_agent
    worker = ToolInteractiveReflectionAgentWorker.from_defaults(
        critique_agent_worker=mock_critique_agent_worker,
        critique_template="mock critique template",
        correction_llm=dummy_llm,
        stopping_callable=mock_stopping_callable,
    )
    # messages that would be sent from the introspective_agent when it delegates
    # to reflection task
    messages = [
        ChatMessage(content="You are a helpful assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="What's 2+2?", role=MessageRole.USER),
        ChatMessage(content="I think it's 5.", role=MessageRole.ASSISTANT),
    ]
    agent = worker.as_agent(chat_history=messages)

    # Act
    response = agent.chat("I think it's 5.")  # reflect on current response

    # Assert
    if PRINT_CHAT_HISTORY:
        for msg in agent.chat_history:
            print(str(msg))
            print()
    assert response.response == "This is a mock correction."
    assert (
        len(agent.chat_history) == 8
    )  # (system, user, asst, user, ref, cor, ref, asst)


@patch("llama_index.core.agent.function_calling.step.FunctionCallingAgentWorker")
def test_introspective_agent_with_max_iterations(mock_critique_agent_worker) -> None:
    # Arrange
    mock_critique_agent = MagicMock()
    mock_critique_agent.chat.side_effect = [
        AgentChatResponse(response="This is a mock critique str."),
        AgentChatResponse(response="This is another mock critique str."),
    ]
    mock_stopping_callable = MagicMock()
    mock_stopping_callable.side_effect = [False, True]
    dummy_llm = MockLLM()

    mock_critique_agent_worker.as_agent.return_value = mock_critique_agent
    worker = ToolInteractiveReflectionAgentWorker.from_defaults(
        critique_agent_worker=mock_critique_agent_worker,
        critique_template="mock critique template",
        correction_llm=dummy_llm,
        max_iterations=1,
    )
    # messages that would be sent from the introspective_agent when it delegates
    # to reflection task
    messages = [
        ChatMessage(content="You are a helpful assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="What's 2+2?", role=MessageRole.USER),
        ChatMessage(content="I think it's 5.", role=MessageRole.ASSISTANT),
    ]
    agent = worker.as_agent(chat_history=messages)

    # Act
    response = agent.chat("I think it's 5.")  # reflect on current response

    # Assert
    if PRINT_CHAT_HISTORY:
        for msg in agent.chat_history:
            print(str(msg))
            print()
    assert response.response == "This is a mock correction."
    assert len(agent.chat_history) == 6  # (system, user, asst, user, ref, cor/asst)
