import pytest

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.indices.prompt_helper import ChatPromptHelper
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
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

    @pytest.mark.asyncio
    async def test_get_response_from_messages__empty_chunks(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = SimpleSummarize(llm=llm, multimodal=True)

        sync_resp = synthesizer.get_response_from_messages(
            query_str="test", message_chunks=[]
        )
        assert sync_resp == " ".join(["text"] * 10)

        async_resp = await synthesizer.aget_response_from_messages(
            query_str="test", message_chunks=[]
        )
        assert async_resp == " ".join(["text"] * 10)

    def test_strict_truncation_propagated_to_chat_prompt_helper(
        self, png_1px_b64: bytes
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        default_synthesizer = SimpleSummarize(
            llm=llm, multimodal=True, strict_truncation=True
        )
        assert default_synthesizer._chat_prompt_helper._strict_truncation is True

        chat_prompt_helper = ChatPromptHelper(
            context_window=300, num_output=10, strict_truncation=False
        )
        synthesizer = SimpleSummarize(
            llm=llm,
            chat_prompt_helper=chat_prompt_helper,
            multimodal=True,
            strict_truncation=True,
        )
        nodes = [NodeWithScore(node=ImageNode(image=png_1px_b64), score=1.0)]
        synthesizer.synthesize(query="test", nodes=nodes)
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]
        assert chat_prompt_helper._strict_truncation is False
