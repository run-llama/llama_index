import pytest

from llama_index.core.base.response.schema import (
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.response_synthesizers.generation import (
    Generation,
    MultimodalGeneration,
)


class TestGeneration:
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


class TestMultimodalGeneration:
    def test_init__non_chat_model_raises_error(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10)
        with pytest.raises(
            ValueError, match="BaseMultimodalSynthesizer requires a chat LLM."
        ):
            MultimodalGeneration(llm=llm)

    def test_synthesize(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalGeneration(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=[])
        assert isinstance(response, Response)
        assert str(response) == "user: test\nassistant: "
        assert llm.last_called_chat_function == ["chat"]
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]

    def test_synthesize__streaming(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalGeneration(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=[])
        assert isinstance(response, StreamingResponse)
        assert str(response) == "user: test\nassistant: "
        assert llm.last_called_chat_function == ["stream_chat"]
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]

    @pytest.mark.asyncio
    async def test_asynthesize(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalGeneration(llm=llm)
        response = await synthesizer.asynthesize(query="test", nodes=[])
        assert isinstance(response, Response)
        assert str(response) == "user: test\nassistant: "
        assert set(llm.last_called_chat_function) == {"achat", "chat"}, (
            "Async calls sync under hood"
        )
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]

    @pytest.mark.asyncio
    async def test_asynthesize__streaming(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(is_chat_model=True)
        synthesizer = MultimodalGeneration(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=[])
        assert isinstance(response, AsyncStreamingResponse)
        assert str(response) == "user: test\nassistant: "
        assert set(llm.last_called_chat_function) == {"stream_chat", "astream_chat"}, (
            "Async calls sync under hood"
        )
        assert len(llm.last_chat_messages) == 1, "Only user messages should be present"
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text"
        ]
