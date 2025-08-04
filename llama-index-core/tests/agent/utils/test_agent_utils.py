import pytest
from typing import Any, List, Type, Optional, Dict
from typing_extensions import override

from pydantic import BaseModel
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.types import Model
from llama_index.core.llms import (
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    LLM,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.tools import ToolSelection
from llama_index.core.agent.utils import (
    messages_to_xml_format,
    generate_structured_response,
)


class Structure(BaseModel):
    hello: str
    world: int


class TestLLM(LLM):
    def __init__(self, responses: List[ChatMessage], structured_response: str):
        super().__init__()
        self._responses = responses
        self._structured_response = structured_response
        self._response_index = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])

    @override
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls.model_validate_json(self._structured_response)

    @override
    async def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls.model_validate_json(self._structured_response)

    async def achat(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        pass

    def stream_chat(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    async def acomplete(self, *args, **kwargs):
        pass

    def stream_complete(self, *args, **kwargs):
        pass

    async def astream_complete(self, *args, **kwargs):
        pass

    def _prepare_chat_with_tools(self, *args, **kwargs):
        return {}


@pytest.fixture
def chat_messages() -> List[ChatMessage]:
    return [
        ChatMessage(role="user", blocks=[TextBlock(text="hello")]),
        ChatMessage(role="assistant", blocks=[TextBlock(text="hello back")]),
        ChatMessage(role="user", blocks=[TextBlock(text="how are you?")]),
        ChatMessage(role="assistant", blocks=[TextBlock(text="I am good, thank you.")]),
    ]


@pytest.fixture()
def chat_messages_sys(chat_messages: List[ChatMessage]) -> List[ChatMessage]:
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        *chat_messages,
    ]


@pytest.fixture
def xml_string() -> str:
    return "<current_conversation>\n\t<user>\n\t\t<message>hello</message>\n\t</user>\n\t<assistant>\n\t\t<message>hello back</message>\n\t</assistant>\n\t<user>\n\t\t<message>how are you?</message>\n\t</user>\n\t<assistant>\n\t\t<message>I am good, thank you.</message>\n\t</assistant>\n</current_conversation>\n\nGiven the conversation, format the output according to the provided schema."


@pytest.fixture
def xml_string_sys() -> str:
    return "<current_conversation>\n\t<system>\n\t\t<message>You are a helpful assistant.</message>\n\t</system>\n\t<user>\n\t\t<message>hello</message>\n\t</user>\n\t<assistant>\n\t\t<message>hello back</message>\n\t</assistant>\n\t<user>\n\t\t<message>how are you?</message>\n\t</user>\n\t<assistant>\n\t\t<message>I am good, thank you.</message>\n\t</assistant>\n</current_conversation>\n\nGiven the conversation, format the output according to the provided schema."


@pytest.fixture
def structured_response() -> str:
    return Structure(hello="test", world=1).model_dump_json()


def test_messages_to_xml(chat_messages: List[ChatMessage], xml_string: str) -> None:
    msg = messages_to_xml_format(chat_messages)
    assert len(msg) == 1
    assert isinstance(msg[0], ChatMessage)
    s = ""
    for block in msg[0].blocks:
        s += block.text
    assert s == xml_string


def test_messages_to_xml_sys(
    chat_messages_sys: List[ChatMessage], xml_string_sys: str
) -> None:
    msg = messages_to_xml_format(chat_messages_sys)
    assert len(msg) == 2
    assert isinstance(msg[0], ChatMessage)
    assert msg[0].role == "system"
    assert msg[0].content == "You are a helpful assistant."
    s = ""
    for block in msg[1].blocks:
        s += block.text
    assert s == xml_string_sys


@pytest.mark.asyncio
async def test_generate_structured_response(
    chat_messages: List[ChatMessage], structured_response: str
) -> None:
    llm = TestLLM(
        responses=[ChatMessage(role="assistant", content="Hello World!")],
        structured_response=structured_response,
    )
    generated_response = await generate_structured_response(
        messages=chat_messages, llm=llm, output_cls=Structure
    )
    assert Structure.model_validate(
        generated_response
    ) == Structure.model_validate_json(structured_response)
