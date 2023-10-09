"""Test LLM program."""

import json
from unittest.mock import MagicMock

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.program.llm_program import LLMTextCompletionProgram
from llama_index.prompts import ChatPromptTemplate


class MockLLM(MagicMock):
    def complete(self, prompt: str) -> CompletionResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return CompletionResponse(text=text)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()


class MockChatLLM(MagicMock):
    def chat(self, prompt: str) -> ChatResponse:
        test_object = {"hello": "chat"}
        text = json.dumps(test_object)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text)
        )

    @property
    def metadata(self) -> LLMMetadata:
        metadata = LLMMetadata()
        metadata.is_chat_model = True
        return metadata


class TestModel(BaseModel):
    __test__ = False
    hello: str


def test_llm_program() -> None:
    """Test LLM program."""
    output_parser = PydanticOutputParser(output_cls=TestModel)
    llm_program = LLMTextCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt_template_str="This is a test prompt with a {test_input}.",
        llm=MockLLM(),
    )
    # mock llm
    obj_output = llm_program(test_input="hello")
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"


def test_llm_program_with_messages() -> None:
    """Test LLM program."""
    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    prompt = ChatPromptTemplate(message_templates=messages)
    output_parser = PydanticOutputParser(output_cls=TestModel)
    llm_program = LLMTextCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt=prompt,
        llm=MockLLM(),
    )
    # mock llm
    obj_output = llm_program()
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "world"


def test_llm_program_with_messages_and_chat() -> None:
    """Test LLM program."""
    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    prompt = ChatPromptTemplate(message_templates=messages)
    output_parser = PydanticOutputParser(output_cls=TestModel)
    llm_program = LLMTextCompletionProgram.from_defaults(
        output_parser=output_parser,
        prompt=prompt,
        llm=MockChatLLM(),
    )
    # mock llm
    obj_output = llm_program()
    assert isinstance(obj_output, TestModel)
    assert obj_output.hello == "chat"
