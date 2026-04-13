import pytest

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.response_synthesizers.simple_summarize import (
    MultimodalSimpleSummarize,
    SimpleSummarize,
)
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


class TestMultimodalSimpleSummarize:
    def test_init__non_chat_model_raises_error(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10)
        with pytest.raises(
            ValueError, match="BaseMultimodalSynthesizer requires a chat LLM."
        ):
            MultimodalSimpleSummarize(llm=llm)

    def test_synthesize(self, multimodal_nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalSimpleSummarize(llm=llm)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
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

    def test_synthesize__streaming(self, multimodal_nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalSimpleSummarize(llm=llm, streaming=True)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, StreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == ["stream_chat"]
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
    async def test_asynthesize(self, multimodal_nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalSimpleSummarize(llm=llm)
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
    async def test_asynthesize__streaming(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = MultimodalSimpleSummarize(llm=llm, streaming=True)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        assert isinstance(response, AsyncStreamingResponse)
        assert str(response) == " ".join(["text"] * 10)
        assert len(llm.last_called_chat_function) == 2
        assert set(llm.last_called_chat_function) == {"stream_chat", "astream_chat"}, (
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
