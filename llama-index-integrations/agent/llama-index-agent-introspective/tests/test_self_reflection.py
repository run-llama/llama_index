from typing import Any

from collections import deque
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.prompts.base import PromptTemplate

from llama_index.agent.introspective.reflective.self_reflection import (
    Reflection,
    Correction,
    SelfReflectionAgentWorker,
)

PRINT_CHAT_HISTORY = False

mock_reflections_queue = deque(
    [
        Reflection(is_done=False, feedback="This is a mock reflection."),
        Reflection(is_done=True, feedback="This is a mock reflection."),
    ]
)
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
        if output_cls == Reflection:
            return mock_reflections_queue.popleft()
        elif output_cls == Correction:
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


def test_reflection_agent() -> None:
    # Arrange
    dummy_llm = MockLLM()
    worker = SelfReflectionAgentWorker.from_defaults(llm=dummy_llm)
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
