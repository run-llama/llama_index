import pytest

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.indices.prompt_helper import ChatPromptHelper
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.response_synthesizers.simple_summarize import SimpleSummarize
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode


@pytest.fixture()
def nodes() -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text="context information"), score=1.0)]


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def multimodal_nodes(png_1px_b64) -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="context information"), score=1.0),
        NodeWithScore(node=ImageNode(image=png_1px_b64), score=0.9),
    ]


class TestSimpleSummarize:
    def test_init__multimodal_with_non_chat_model_raises_error(self) -> None:
        # Arrange
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10)

        # Act / Assert
        with pytest.raises(
            ValueError, match="Multimodal synthesis requires a chat LLM."
        ):
            SimpleSummarize(llm=llm, multimodal=True)

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    def test_synthesize(self, llm, nodes: list[NodeWithScore]) -> None:
        synthesizer = SimpleSummarize(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        if llm.metadata.is_chat_model:
            assert llm.last_called_chat_function == ["chat"]
            assert [msg.role for msg in llm.last_chat_messages] == [
                MessageRole.SYSTEM,
                MessageRole.USER,
            ]
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    def test_synthesize__streaming(self, llm, nodes: list[NodeWithScore]) -> None:
        synthesizer = SimpleSummarize(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=nodes)
        assert isinstance(response, StreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        if llm.metadata.is_chat_model:
            assert llm.last_called_chat_function == ["stream_chat"]
            assert [msg.role for msg in llm.last_chat_messages] == [
                MessageRole.SYSTEM,
                MessageRole.USER,
            ]
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        # Arrange
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = SimpleSummarize(llm=llm, multimodal=True)

        # Act
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)

        # Assert
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == ["chat"]
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ]

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize(self, llm, nodes: list[NodeWithScore]) -> None:
        synthesizer = SimpleSummarize(llm=llm)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        if llm.metadata.is_chat_model:
            assert len(llm.last_called_chat_function) == 2
            assert set(llm.last_called_chat_function) == {"chat", "achat"}, (
                "Async calls sync under hood"
            )
            assert [msg.role for msg in llm.last_chat_messages] == [
                MessageRole.SYSTEM,
                MessageRole.USER,
            ]
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize__streaming(
        self, llm, nodes: list[NodeWithScore]
    ) -> None:
        synthesizer = SimpleSummarize(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert isinstance(response, AsyncStreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        if llm.metadata.is_chat_model:
            assert len(llm.last_called_chat_function) == 2
            assert set(llm.last_called_chat_function) == {
                "stream_chat",
                "astream_chat",
            }, "Async calls sync under hood"
            assert [msg.role for msg in llm.last_chat_messages] == [
                MessageRole.SYSTEM,
                MessageRole.USER,
            ]
        else:
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = SimpleSummarize(llm=llm, multimodal=True)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, Response)
        assert str(response) == " ".join(["text"] * 10)
        assert len(llm.last_called_chat_function) == 2
        assert set(llm.last_called_chat_function) == {"chat", "achat"}, (
            "Async calls sync under hood"
        )
        assert [msg.role for msg in llm.last_chat_messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("different_field", ["role", "additional_kwargs"])
def test_get_response_preserves_unmerged_messages(
    streaming: bool, different_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "_tokenizer", lambda text: list(text))
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
    synthesizer = SimpleSummarize(llm=llm, multimodal=True, streaming=streaming)
    second_kwargs = (
        {"role": MessageRole.ASSISTANT}
        if different_field == "role"
        else {"additional_kwargs": {"source": "second"}}
    )
    chunks = [
        ChatMessage(content="first source"),
        ChatMessage(content="second unique source", **second_kwargs),
    ]

    response = synthesizer.get_response_from_messages("test", chunks)
    if streaming:
        list(response)

    prompt = "\n".join(message.content or "" for message in llm.last_chat_messages)
    assert "first source" in prompt
    assert "second unique source" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("different_field", ["role", "additional_kwargs"])
async def test_aget_response_preserves_unmerged_messages(
    streaming: bool, different_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "_tokenizer", lambda text: list(text))
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
    synthesizer = SimpleSummarize(llm=llm, multimodal=True, streaming=streaming)
    second_kwargs = (
        {"role": MessageRole.ASSISTANT}
        if different_field == "role"
        else {"additional_kwargs": {"source": "second"}}
    )
    chunks = [
        ChatMessage(content="first source"),
        ChatMessage(content="second unique source", **second_kwargs),
    ]

    response = await synthesizer.aget_response_from_messages("test", chunks)
    if streaming:
        async for _ in response:
            pass

    prompt = "\n".join(message.content or "" for message in llm.last_chat_messages)
    assert "first source" in prompt
    assert "second unique source" in prompt


def test_unmerged_messages_share_context_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Settings, "_tokenizer", lambda text: list(text))
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
    helper = ChatPromptHelper(context_window=700, num_output=10)
    synthesizer = SimpleSummarize(llm=llm, multimodal=True, chat_prompt_helper=helper)
    chunks = [
        ChatMessage(content="a" * 200),
        ChatMessage(content="b" * 200, role=MessageRole.ASSISTANT),
    ]

    synthesizer.get_response_from_messages("test", chunks)

    context = llm.last_chat_messages[-1].content
    assert "a" * 20 in context
    assert "b" * 20 in context
    prompt_tokens = TokenCounter().estimate_tokens_in_messages(llm.last_chat_messages)
    assert prompt_tokens + helper.num_output <= helper.context_window


def test_get_response_from_empty_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "_tokenizer", lambda text: list(text))
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
    synthesizer = SimpleSummarize(llm=llm, multimodal=True)

    assert synthesizer.get_response_from_messages("test", []) == " ".join(["text"] * 10)


@pytest.mark.asyncio
async def test_aget_response_from_empty_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Settings, "_tokenizer", lambda text: list(text))
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
    synthesizer = SimpleSummarize(llm=llm, multimodal=True)

    assert await synthesizer.aget_response_from_messages("test", []) == " ".join(
        ["text"] * 10
    )
