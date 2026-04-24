import pytest

from llama_index.core.base.response.schema import (
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.response_synthesizers.generation import Generation
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode


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


class TestGeneration:
    def test_init__multimodal_with_non_chat_model_raises_error(self) -> None:
        # Arrange
        llm = MockLLMWithChatMemoryOfLastCall()

        # Act / Assert
        with pytest.raises(
            ValueError, match="Multimodal synthesis requires a chat LLM."
        ):
            Generation(llm=llm, multimodal=True)

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(),
            MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
        ],
    )
    def test_synthesize(self, llm) -> None:
        synthesizer = Generation(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=[])
        assert isinstance(response, Response)
        if llm.metadata.is_chat_model:
            assert str(response) == "user: test\nassistant: "
            assert llm.last_called_chat_function == ["chat"]
            assert len(llm.last_chat_messages) == 1, (
                "Only user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
        else:
            assert str(response) == "test"
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(),
            MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
        ],
    )
    def test_synthesize__streaming(self, llm) -> None:
        synthesizer = Generation(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=[])
        assert isinstance(response, StreamingResponse)
        if llm.metadata.is_chat_model:
            assert str(response) == "user: test\nassistant: "
            assert llm.last_called_chat_function == ["stream_chat"]
            assert len(llm.last_chat_messages) == 1, (
                "Only user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
        else:
            assert str(response) == "test"
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    def test_synthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        # Arrange
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = Generation(llm=llm, multimodal=True)

        # Act
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)

        # Assert
        assert isinstance(response, Response)
        assert str(response) == "user: test\nassistant: "
        assert llm.last_called_chat_function == ["chat"]
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(),
            MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize(self, llm) -> None:
        synthesizer = Generation(llm=llm)
        response = await synthesizer.asynthesize(query="test", nodes=[])
        assert isinstance(response, Response)
        if llm.metadata.is_chat_model:
            assert str(response) == "user: test\nassistant: "
            assert set(llm.last_called_chat_function) == {"chat", "achat"}, (
                "Async calls sync under hood"
            )
            assert len(llm.last_chat_messages) == 1, (
                "Only user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
        else:
            assert str(response) == "test"
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(),
            MockLLMWithChatMemoryOfLastCall(is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize__streaming(self, llm) -> None:
        synthesizer = Generation(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=[])
        assert isinstance(response, AsyncStreamingResponse)
        if llm.metadata.is_chat_model:
            assert str(response) == "user: test\nassistant: "
            assert set(llm.last_called_chat_function) == {
                "astream_chat",
                "stream_chat",
            }, "Async calls sync under hood"
            assert len(llm.last_chat_messages) == 1, (
                "Only user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
        else:
            assert str(response) == "test"
            assert llm.last_called_chat_function == []
            assert llm.last_chat_messages is None

    @pytest.mark.asyncio
    async def test_asynthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        # Arrange
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = Generation(llm=llm, multimodal=True)

        # Act
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)

        # Assert
        assert isinstance(response, Response)
        assert str(response) == "user: test\nassistant: "
        assert set(llm.last_called_chat_function) == {"chat", "achat"}, (
            "Async calls sync under hood"
        )
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]
