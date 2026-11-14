"""Test LLM program."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.program.llm_program import LLMTextCompletionProgram
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate


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


class StreamingLLM(CustomLLM):
    response: str
    chunk_size: int
    is_chat_model: bool

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=self.is_chat_model)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=self.response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        for start in range(0, len(self.response), self.chunk_size):
            end = start + self.chunk_size
            yield CompletionResponse(
                text=self.response[:end], delta=self.response[start:end]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_chat_model", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 1000])
@pytest.mark.parametrize(
    ("opening", "closing"),
    [("", ""), ("```json\n", "\n```"), (" \r\n```\r\n", "\r\n``` \r\n")],
    ids=["plain", "json-fence", "bare-fence-crlf"],
)
async def test_stream_structured_predict_fenced_json(
    is_async: bool, is_chat_model: bool, chunk_size: int, opening: str, closing: str
) -> None:
    class Response(BaseModel):
        items: list[TestModel]

    expected = Response(
        items=[TestModel(hello="keep ```json and ```"), TestModel(hello="world")]
    )
    llm = StreamingLLM(
        response=f"{opening}{expected.model_dump_json()}{closing}",
        chunk_size=chunk_size,
        is_chat_model=is_chat_model,
    )
    prompt = PromptTemplate("Return the items.")
    if is_async:
        stream = await llm.astream_structured_predict(Response, prompt)
        outputs = [output async for output in stream]
    else:
        outputs = list(llm.stream_structured_predict(Response, prompt))

    assert outputs[-1] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_stream_structured_predict_fenced_partial_json(is_async: bool) -> None:
    llm = StreamingLLM(
        response='```json\n{"hello": "keep ```json and ```"}\n```',
        chunk_size=1,
        is_chat_model=False,
    )
    prompt = PromptTemplate("Return a greeting.")
    if is_async:
        stream = await llm.astream_structured_predict(TestModel, prompt)
        outputs = [output async for output in stream]
    else:
        outputs = list(llm.stream_structured_predict(TestModel, prompt))

    assert any(output.hello == "keep ```" for output in outputs)
    assert outputs[-1] == TestModel(hello="keep ```json and ```")


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
