from collections import OrderedDict
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Type,
    cast,
    Sequence,
)
from unittest.mock import patch, MagicMock

import pytest
import tiktoken

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ToolCallBlock,
    MessageRole,
    TextBlock,
    ImageBlock,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import (
    MockLLM,
    MockLLMWithChatMemoryOfLastCall,
    MockFunctionCallingLLMWithChatMemoryOfLastCall,
)
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.response_synthesizers.refine import (
    MultimodalRefine,
    Refine,
    StructuredRefineResponse,
    DefaultRefineProgram,
)
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode
from llama_index.core.types import BasePydanticProgram


class FailingStub(BasePydanticProgram):
    """Stub that always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(self, *args: Any, **kwargs: Any) -> StructuredRefineResponse:
        raise self._exc


class QuerySatisfiedCase(BaseModel):
    input2_value: bool
    expected_response: str
    expected_last_llm_message_prefix: str


class LLMCase(BaseModel):
    llm: MockLLMWithChatMemoryOfLastCall
    chat_function: str | None = None
    async_chat_function: str | None = None
    streaming_chat_function: str | None = None
    async_streaming_chat_function: str | None = None


@pytest.fixture()
def nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="input1"), score=1.0),
        NodeWithScore(node=TextNode(text="input2"), score=0.9),
        NodeWithScore(node=TextNode(text="input3"), score=0.8),
    ]


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def multimodal_nodes(png_1px_b64: bytes) -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="input1"), score=1.0),
        NodeWithScore(node=TextNode(text="input2"), score=0.9),
        NodeWithScore(node=ImageNode(image=png_1px_b64), score=0.8),
    ]


@pytest.fixture(
    params=[
        pytest.param(
            LLMCase(
                llm=MockLLMWithChatMemoryOfLastCall(),
            ),
            id="non chat model",
        ),
        pytest.param(
            LLMCase(
                llm=MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
                chat_function="chat",
                async_chat_function="achat",
                streaming_chat_function="stream_chat",
                async_streaming_chat_function="astream_chat",
            ),
            id="chat model",
        ),
    ]
)
def llm_case(request) -> MockLLMWithChatMemoryOfLastCall:
    return request.param


@pytest.fixture()
def input_to_query_satisfied(png_1px_b64) -> OrderedDict[str | None, bool]:
    return OrderedDict(
        [
            ("input1", False),
            ("input2", True),
            ("input3", False),
            (ImageBlock(image=png_1px_b64).inline_url(), False),
            (None, False),  # In the MockRefineProgram, binary block content is None
        ]
    )


@pytest.fixture(
    params=[
        pytest.param(
            QuerySatisfiedCase(
                input2_value=True,
                expected_response="input2",
                expected_last_llm_message_prefix=(
                    # Refine template is used after first query satisfied results in a temporary answer
                    "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
                ),
            ),
            id="one query satisfied",
        ),
        pytest.param(
            QuerySatisfiedCase(
                input2_value=False,
                expected_response="Empty Response",
                expected_last_llm_message_prefix=(
                    # QA template is repeated until the first query_satisfied = True
                    # when all query_satisfied values are False, we never get to refining
                    "You are an expert Q&A system that is trusted around the world"
                ),
            ),
            id="no queries satisfied",
        ),
    ]
)
def query_satisfied_case(request) -> QuerySatisfiedCase:
    return request.param


def extract_input_context_from_messages(
    messages_or_str: Sequence[ChatMessage] | str,
) -> str:
    if isinstance(messages_or_str, str):
        content = messages_or_str
        return content.split("---\n")[1].split("\n\n---")[0].split("\n---")[0]
    else:
        messages = messages_or_str
        if (
            len(messages) == 2
        ):  # CHAT_TEXT_QA_PROMPT, CHAT_CONTENT_QA_PROMPT and CHAT_CONTENT_REFINE_PROMPT
            if len(messages[1].blocks) == 1:  # All text
                content = messages[1].content
                input_str = cast(
                    str, content.split("---\n")[1].split("\n\n---")[0].split("\n---")[0]
                )
            else:  # binary block as context
                input_str = messages[1].blocks[1].inline_url()
        else:  # CHAT_REFINE_PROMPT
            content = cast(str, messages[0].content)
            input_str = content.split("New Context: ")[1].split("\n")[0]
    return input_str


def tool_call_json_from_messages(
    messages_or_str: Sequence[ChatMessage] | str,
    input_to_query_satisfied: OrderedDict[str | None, bool],
) -> str:
    input_str = extract_input_context_from_messages(messages_or_str)
    query_satisfied = input_to_query_satisfied[input_str]
    return " ".join(
        StructuredRefineResponse(query_satisfied=query_satisfied, answer=input_str)
        .model_dump_json(indent=4)
        .split()
    )


@pytest.fixture()
def mock_chat_message_with_tool_call_generator(
    input_to_query_satisfied,
) -> Callable[[Sequence[ChatMessage]], ChatMessage]:
    def func(messages: Sequence[ChatMessage]) -> ChatMessage:
        tool_args_json = tool_call_json_from_messages(
            messages, input_to_query_satisfied
        )
        return ChatMessage(
            blocks=[
                ToolCallBlock(
                    block_type="tool_call",
                    tool_call_id="call_abc123",
                    tool_name="StructuredRefineResponse",
                    tool_kwargs=tool_args_json,
                )
            ],
            role=MessageRole.ASSISTANT,
        )

    return func


@pytest.fixture()
def mock_streaming_chat_message_with_tool_call_generator(
    input_to_query_satisfied,
) -> Callable[[Sequence[ChatMessage]], Generator[ChatMessage, None, None]]:
    def func(messages: Sequence[ChatMessage]) -> Generator[ChatMessage, None, None]:
        tool_args_json = tool_call_json_from_messages(
            messages, input_to_query_satisfied
        )
        return (
            ChatMessage(
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="call_abc123",
                            tool_name="StructuredRefineResponse",
                            tool_kwargs=parse_partial_json(
                                " ".join(tool_args_json.split()[:i]) or "{"
                            ),
                        )
                    ]
                },
                blocks=[
                    TextBlock(text=""),
                    ToolCallBlock(
                        block_type="tool_call",
                        tool_call_id="call_abc123",
                        tool_name="StructuredRefineResponse",
                        tool_kwargs=" ".join(tool_args_json.split()[:i]) or "{",
                    ),
                ],
                role=MessageRole.ASSISTANT,
            )
            for i in range(len(tool_args_json.split()) + 1)
        )

    return func


@pytest.fixture()
def mock_async_streaming_chat_message_with_tool_call_generator(
    input_to_query_satisfied, mock_streaming_chat_message_with_tool_call_generator
) -> Coroutine[None, None, AsyncGenerator[ChatMessage, None]]:
    async def coro(
        messages: Sequence[ChatMessage],
    ) -> AsyncGenerator[ChatMessage, None]:
        for chat_message in mock_streaming_chat_message_with_tool_call_generator(
            messages
        ):
            yield chat_message

    return coro


@pytest.fixture()
def mock_completion_response_text_completion_generator(
    input_to_query_satisfied: OrderedDict[str | None, bool],
) -> Callable[[Sequence[ChatMessage]], CompletionResponse]:
    def func(messages, **kwargs) -> CompletionResponse:
        tool_args_json = tool_call_json_from_messages(
            messages, input_to_query_satisfied
        )
        return CompletionResponse(text=tool_args_json)

    return func


@pytest.fixture()
def mock_chat_response_text_completion_generator(
    mock_completion_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], CompletionResponse
    ],
) -> Callable[[Sequence[ChatMessage]], ChatResponse]:
    def func(messages, **kwargs) -> ChatResponse:
        return completion_response_to_chat_response(
            mock_completion_response_text_completion_generator(messages)
        )

    return func


@pytest.fixture()
def mock_async_completion_response_text_completion_generator(
    mock_completion_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], CompletionResponse
    ],
) -> Coroutine[Any, Any, CompletionResponse]:
    async def coro(messages, **kwargs) -> CompletionResponse:
        return mock_completion_response_text_completion_generator(messages)

    return coro


@pytest.fixture()
def mock_async_chat_response_text_completion_generator(
    mock_completion_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], CompletionResponse
    ],
) -> Coroutine[Any, Any, ChatResponse]:
    async def coro(messages, **kwargs) -> ChatResponse:
        return completion_response_to_chat_response(
            mock_completion_response_text_completion_generator(messages)
        )

    return coro


@pytest.fixture()
def mock_streaming_completion_response_text_completion_generator(
    input_to_query_satisfied: OrderedDict[str | None, bool],
) -> Callable[[Sequence[ChatMessage]], Generator[CompletionResponse, None, None]]:
    def func(messages, **kwargs) -> Generator[CompletionResponse, None, None]:
        tool_args_json = tool_call_json_from_messages(
            messages, input_to_query_satisfied
        )
        text = ""
        encoder = tiktoken.get_encoding("cl100k_base")
        chunks = [
            encoder.decode_single_token_bytes(tkn).decode()
            for tkn in encoder.encode(tool_args_json)
        ]
        for chunk in chunks:
            text += chunk
            yield CompletionResponse(text=text, delta=chunk)

    return func


@pytest.fixture()
def mock_streaming_chat_response_text_completion_generator(
    mock_streaming_completion_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], Generator[CompletionResponse, None, None]
    ],
) -> Callable[[Sequence[ChatMessage]], Generator[ChatResponse, None, None]]:
    def func(messages, **kwargs) -> Generator[ChatResponse, None, None]:
        for (
            completion_resp
        ) in mock_streaming_completion_response_text_completion_generator(messages):
            chat_response = completion_response_to_chat_response(completion_resp)
            chat_response.delta = completion_resp.delta
            yield chat_response

    return func


@pytest.fixture()
def mock_async_streaming_completion_response_text_completion_generator(
    mock_streaming_completion_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], Generator[CompletionResponse, None, None]
    ],
) -> Coroutine[Any, Any, AsyncGenerator[CompletionResponse, None]]:
    async def coro(messages, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        for (
            completion_resp
        ) in mock_streaming_completion_response_text_completion_generator(messages):
            yield completion_resp

    return coro


@pytest.fixture()
def mock_async_streaming_chat_response_text_completion_generator(
    mock_streaming_chat_response_text_completion_generator: Callable[
        [Sequence[ChatMessage]], Generator[ChatResponse, None, None]
    ],
) -> Coroutine[Any, Any, AsyncGenerator[ChatResponse, None]]:
    async def coro(messages, **kwargs) -> AsyncGenerator[ChatResponse, None]:
        async def gen():
            for chat_response in mock_streaming_chat_response_text_completion_generator(
                messages
            ):
                yield chat_response

        return gen()

    return coro


class TestRefine:
    def test_constructor_args(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        with pytest.raises(ValueError):
            # can't construct refine with a program factory but not answer filtering
            Refine(
                llm=llm,
                program_factory=lambda _: DefaultRefineProgram(
                    prompt=MagicMock(), llm=llm, output_cls=MagicMock()
                ),
                structured_answer_filtering=False,
            )

    def test_synthesize__default_refine_program(
        self, llm_case: LLMCase, nodes: list[NodeWithScore]
    ) -> None:
        llm = llm_case.llm
        llm.max_tokens = 10
        synthesizer = Refine(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        if llm_case.chat_function is not None:
            assert llm.last_called_chat_function.count(llm_case.chat_function) == len(
                nodes
            )
            assert llm.last_chat_messages[0].content.startswith(
                "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__default_refine_program(
        self, llm_case: LLMCase, nodes: list[NodeWithScore]
    ) -> None:
        llm = llm_case.llm
        llm.max_tokens = 10
        synthesizer = Refine(llm=llm)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        if llm_case.async_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.async_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__streaming_default_refine_program(
        self, llm_case: LLMCase, nodes: list[NodeWithScore]
    ) -> None:
        llm = llm_case.llm
        llm.max_tokens = 10
        synthesizer = Refine(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, StreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        if llm_case.streaming_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.streaming_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__streaming_default_refine_program(
        self, llm_case: LLMCase, nodes: list[NodeWithScore]
    ) -> None:
        llm = llm_case.llm
        llm.max_tokens = 10
        synthesizer = Refine(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, AsyncStreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        if llm_case.async_streaming_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.async_streaming_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__structured_answer_filtering_default_text_completion_refine_program(
        self,
        llm_case: LLMCase,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_completion_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], CompletionResponse
        ],
        mock_chat_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], ChatResponse
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = llm_case.llm
        llm.reset_memory()
        synthesizer = Refine(llm=llm, structured_answer_filtering=True)
        with (
            patch.object(
                MockLLM,
                "complete",
                side_effect=mock_completion_response_text_completion_generator,
            ),
            patch.object(
                CustomLLM,
                "chat",
                side_effect=mock_chat_response_text_completion_generator,
            ),
        ):
            response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        if llm_case.chat_function is not None:
            assert llm.last_called_chat_function.count(llm_case.chat_function) == len(
                nodes
            )
            assert llm.last_chat_messages[0].content.startswith(
                query_satisfied_case.expected_last_llm_message_prefix
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_default_text_completion_refine_program(
        self,
        llm_case: LLMCase,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_completion_response_text_completion_generator: Coroutine[
            Any, Any, CompletionResponse
        ],
        mock_async_chat_response_text_completion_generator: Coroutine[
            Any, Any, ChatResponse
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = llm_case.llm
        llm.reset_memory()
        synthesizer = Refine(llm=llm, structured_answer_filtering=True)
        with (
            patch.object(
                MockLLM,
                "acomplete",
                side_effect=mock_async_completion_response_text_completion_generator,
            ),
            patch.object(
                CustomLLM,
                "achat",
                side_effect=mock_async_chat_response_text_completion_generator,
            ),
        ):
            response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        if llm_case.async_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.async_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                query_satisfied_case.expected_last_llm_message_prefix
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__structured_answer_filtering_with_streaming_default_text_completion_refine_program(
        self,
        llm_case: LLMCase,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_streaming_completion_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], Generator[CompletionResponse, None, None]
        ],
        mock_streaming_chat_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], Generator[ChatResponse, None, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = llm_case.llm
        llm.reset_memory()
        synthesizer = Refine(llm=llm, structured_answer_filtering=True, streaming=True)
        with (
            patch.object(
                MockLLM,
                "stream_complete",
                side_effect=mock_streaming_completion_response_text_completion_generator,
            ),
            patch.object(
                CustomLLM,
                "stream_chat",
                side_effect=mock_streaming_chat_response_text_completion_generator,
            ),
        ):
            response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response), (
            "Not a generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        if llm_case.streaming_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.streaming_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                query_satisfied_case.expected_last_llm_message_prefix
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_with_streaming_default_text_completion_refine_program(
        self,
        llm_case: LLMCase,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_streaming_completion_response_text_completion_generator: Coroutine[
            Any, Any, AsyncGenerator[CompletionResponse, None]
        ],
        mock_async_streaming_chat_response_text_completion_generator: Coroutine[
            Any, Any, AsyncGenerator[ChatResponse, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = llm_case.llm
        llm.reset_memory()
        synthesizer = Refine(llm=llm, structured_answer_filtering=True, streaming=True)
        with (
            patch.object(
                MockLLM,
                "astream_complete",
                side_effect=mock_async_streaming_completion_response_text_completion_generator,
            ),
            patch.object(
                CustomLLM,
                "astream_chat",
                side_effect=mock_async_streaming_chat_response_text_completion_generator,
            ),
        ):
            response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response), (
            "Not an async generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        if llm_case.async_streaming_chat_function is not None:
            assert llm.last_called_chat_function.count(
                llm_case.async_streaming_chat_function
            ) == len(nodes)
            assert llm.last_chat_messages[0].content.startswith(
                query_satisfied_case.expected_last_llm_message_prefix
            )
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__structured_answer_filtering_default_function_calling_refine_program(
        self,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], ChatMessage
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = Refine(llm=llm, structured_answer_filtering=True)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function.count("chat") == len(nodes)
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_default_function_calling_refine_program(
        self,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], ChatMessage
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = Refine(llm=llm, structured_answer_filtering=True)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function.count("achat") == len(nodes)
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )

    def test_synthesize__structured_answer_filtering_with_streaming_default_function_calling_refine_program(
        self,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_streaming_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], Generator[ChatMessage, None, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_streaming_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = Refine(llm=llm, structured_answer_filtering=True, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response), (
            "Not a generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function.count("stream_chat") == len(nodes)
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_with_streaming_default_function_calling_refine_program(
        self,
        nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_streaming_chat_message_with_tool_call_generator: Coroutine[
            Any, Any, AsyncGenerator[ChatMessage, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_async_streaming_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = Refine(llm=llm, structured_answer_filtering=True, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response), (
            "Not a generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function.count("astream_chat") == len(nodes)
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("LLM did not return any tool calls"),
            TypeError("Expected BaseModel but got str"),
        ],
    )
    @pytest.mark.asyncio
    async def test_synthesize_and_asynthesize__handles_value_and_type_errors_from_program(
        self, nodes, error
    ) -> None:
        synthesizer = Refine(
            structured_answer_filtering=True,
            program_factory=lambda _: FailingStub(error),
        )
        assert str(synthesizer.synthesize("question", nodes)) == "Empty Response"
        assert str(await synthesizer.asynthesize("question", nodes)) == "Empty Response"


class TestMultimodalRefine:
    def test_init__non_chat_model_raises_error(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10)
        with pytest.raises(
            ValueError, match="BaseMultimodalSynthesizer requires a chat LLM."
        ):
            MultimodalRefine(llm=llm)

    def test_constructor_args(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        with pytest.raises(ValueError):
            # can't construct refine with a program factory but not answer filtering
            MultimodalRefine(
                llm=llm,
                program_factory=lambda _: DefaultRefineProgram(
                    prompt=MagicMock(),
                    llm=llm,
                    output_cls=MagicMock(),
                ),
                structured_answer_filtering=False,
            )

    def test_synthesize__default_refine_program(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == ["chat", "chat", "chat"], (
            "One for each node"
        )
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert llm.last_chat_messages[0].content.startswith(
            "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
        ), "Most recent call should use the refine template system message"
        assert len(llm.last_chat_messages[1].blocks) == 3
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    def test_synthesize__streaming_default_refine_program(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, StreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == [
            "stream_chat",
            "stream_chat",
            "stream_chat",
        ], "One for each node"
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert llm.last_chat_messages[0].content.startswith(
            "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
        ), "Most recent call should use the refine template system message"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__default_refine_program(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        assert len(llm.last_called_chat_function) == 6, "Two calls for each node"
        assert llm.last_called_chat_function.count("achat") == 3, "One for each node"
        assert llm.last_called_chat_function.count("chat") == 3, (
            "Async calls sync under hood"
        )
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert llm.last_chat_messages[0].content.startswith(
            "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
        ), "Most recent call should use the refine template system message"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__streaming_default_refine_program(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, AsyncStreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        assert len(llm.last_called_chat_function) == 6, "Two calls for each node"
        assert llm.last_called_chat_function.count("astream_chat") == 3, (
            "One for each node"
        )
        assert llm.last_called_chat_function.count("stream_chat") == 3, (
            "Async calls sync under hood"
        )
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert llm.last_chat_messages[0].content.startswith(
            "You are an expert Q&A system that strictly operates in two modes when refining existing answers:"
        ), "Most recent call should use the refine template system messaage"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    def test_synthesize__structured_answer_filtering_default_text_completion_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_chat_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], ChatResponse
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm, structured_answer_filtering=True)
        with (
            patch.object(
                CustomLLM,
                "chat",
                side_effect=mock_chat_response_text_completion_generator,
            ),
        ):
            response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == ["chat", "chat", "chat"], (
            "One for each node"
        )
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_default_text_completion_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_chat_response_text_completion_generator: Coroutine[
            Any, Any, ChatResponse
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalRefine(llm=llm, structured_answer_filtering=True)
        with (
            patch.object(
                CustomLLM,
                "achat",
                side_effect=mock_async_chat_response_text_completion_generator,
            ),
        ):
            response = await synthesizer.asynthesize(
                query="test", nodes=multimodal_nodes
            )
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == ["achat", "achat", "achat"], (
            "One for each node, no sync call for mock"
        )
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    def test_synthesize__structured_answer_filtering_with_streaming_default_text_completion_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_streaming_chat_response_text_completion_generator: Callable[
            [Sequence[ChatMessage]], Generator[ChatResponse, None, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalRefine(
            llm=llm, structured_answer_filtering=True, streaming=True
        )
        with (
            patch.object(
                CustomLLM,
                "stream_chat",
                side_effect=mock_streaming_chat_response_text_completion_generator,
            ),
        ):
            response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response), (
            "Not a generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == [
            "stream_chat",
            "stream_chat",
            "stream_chat",
        ], "One for each node"
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_with_streaming_default_text_completion_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_streaming_chat_response_text_completion_generator: Coroutine[
            Any, Any, AsyncGenerator[ChatResponse, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalRefine(
            llm=llm, structured_answer_filtering=True, streaming=True
        )
        with (
            patch.object(
                CustomLLM,
                "astream_chat",
                side_effect=mock_async_streaming_chat_response_text_completion_generator,
            ),
        ):
            response = await synthesizer.asynthesize(
                query="test", nodes=multimodal_nodes
            )
        assert isinstance(response, Response), (
            "Not an async generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == [
            "astream_chat",
            "astream_chat",
            "astream_chat",
        ], "One for each node; no sync call for mock"
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    def test_synthesize__structured_answer_filtering_default_function_calling_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], ChatMessage
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = MultimodalRefine(llm=llm, structured_answer_filtering=True)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == ["chat", "chat", "chat"], (
            "One for each node"
        )
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_default_function_calling_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], ChatMessage
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = MultimodalRefine(llm=llm, structured_answer_filtering=True)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == query_satisfied_case.expected_response
        assert len(llm.last_called_chat_function) == 6, "Two calls for each node"
        assert llm.last_called_chat_function.count("achat") == 3, "One for each node"
        assert llm.last_called_chat_function.count("chat") == 3, (
            "Async calls sync under hood"
        )
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    def test_synthesize__structured_answer_filtering_with_streaming_default_function_calling_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_streaming_chat_message_with_tool_call_generator: Callable[
            [Sequence[ChatMessage]], Generator[ChatMessage, None, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_streaming_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = MultimodalRefine(
            llm=llm, structured_answer_filtering=True, streaming=True
        )
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response), (
            "Not a generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == [
            "stream_chat",
            "stream_chat",
            "stream_chat",
        ], "One for each node"
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.asyncio
    async def test_asynthesize__structured_answer_filtering_with_streaming_default_function_calling_refine_program(
        self,
        multimodal_nodes: list[NodeWithScore],
        input_to_query_satisfied: OrderedDict[str | None, bool],
        mock_async_streaming_chat_message_with_tool_call_generator: Coroutine[
            Any, Any, AsyncGenerator[ChatMessage, None]
        ],
        query_satisfied_case: QuerySatisfiedCase,
    ) -> None:
        input_to_query_satisfied["input2"] = query_satisfied_case.input2_value
        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            response_generator=mock_async_streaming_chat_message_with_tool_call_generator,
            is_chat_model=True,
        )
        synthesizer = MultimodalRefine(
            llm=llm, structured_answer_filtering=True, streaming=True
        )
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response), (
            "Not an async generator since the tokens were consumed as part of the last refine call"
        )
        assert str(response) == query_satisfied_case.expected_response
        assert llm.last_called_chat_function == [
            "astream_chat",
            "astream_chat",
            "astream_chat",
        ], (
            "One for each node; MockFunctionCallingLLM astream_chat doesn't call sync under hood"
        )
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert llm.last_chat_messages[0].content.startswith(
            query_satisfied_case.expected_last_llm_message_prefix
        )
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ], "User message should contain image block"

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("LLM did not return any tool calls"),
            TypeError("Expected BaseModel but got str"),
        ],
    )
    @pytest.mark.asyncio
    async def test_synthesize_and_asynthesize__handles_value_and_type_errors_from_program(
        self, multimodal_nodes, error
    ) -> None:
        synthesizer = MultimodalRefine(
            llm=MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
            structured_answer_filtering=True,
            program_factory=lambda _: FailingStub(error),
        )
        assert (
            str(synthesizer.synthesize("question", multimodal_nodes))
            == "Empty Response"
        )
        assert (
            str(await synthesizer.asynthesize("question", multimodal_nodes))
            == "Empty Response"
        )
