import pytest

from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.response_synthesizers.accumulate import Accumulate
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode


@pytest.fixture()
def nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="context information"), score=1.0),
        NodeWithScore(node=TextNode(text="context information2"), score=0.7),
    ]


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def multimodal_nodes(png_1px_b64: bytes) -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="context information"), score=1.0),
        NodeWithScore(node=ImageNode(image=png_1px_b64), score=0.9),
    ]


class TestAccumulate:
    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    def test_synthesize(
        self, llm: MockLLMWithChatMemoryOfLastCall, nodes: list[NodeWithScore]
    ) -> None:
        synthesizer = Accumulate(llm=llm)
        expected = "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        response = synthesizer.synthesize(query="test", nodes=nodes)

        assert str(response) == expected
        if llm.metadata.is_chat_model:
            assert llm.last_called_chat_function == ["chat", "chat"], (
                "Called once per node"
            )
            assert len(llm.last_chat_messages) == 2, (
                "System and user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
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
    def test_synthesize__streaming_raises_error(
        self, llm: MockLLMWithChatMemoryOfLastCall, nodes: list[NodeWithScore]
    ) -> None:
        synthesizer = Accumulate(llm=llm, streaming=True)
        with pytest.raises(ValueError, match="Unable to stream"):
            synthesizer.synthesize(query="test", nodes=nodes)

    def test_synthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = Accumulate(llm=llm, multimodal=True)
        response = synthesizer.synthesize(query="test", nodes=multimodal_nodes)
        expected = "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        assert str(response) == expected
        assert llm.last_called_chat_function == ["chat", "chat"]
        assert len(llm.last_chat_messages) == 2
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
    async def test_asynthesize(
        self, llm: MockLLMWithChatMemoryOfLastCall, nodes: list[NodeWithScore]
    ) -> None:
        synthesizer = Accumulate(llm=llm)
        expected = "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        response = await synthesizer.asynthesize(query="test", nodes=nodes)
        assert str(response) == expected
        if llm.metadata.is_chat_model:
            assert set(llm.last_called_chat_function) == {"chat", "achat"}, (
                "2x Async->Sync calls sync under hood"
            )
            assert len(llm.last_chat_messages) == 2, (
                "System and user messages should be present"
            )
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text"]
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
    async def test_asynthesize__streaming_raises_error(
        self, llm: MockLLMWithChatMemoryOfLastCall, nodes: list[NodeWithScore]
    ) -> None:
        synthesizer = Accumulate(llm=llm, streaming=True)
        with pytest.raises(ValueError, match="Unable to stream"):
            await synthesizer.asynthesize(query="test", nodes=nodes)

    @pytest.mark.asyncio
    async def test_asynthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore]
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer = Accumulate(llm=llm, multimodal=True)
        response = await synthesizer.asynthesize(query="test", nodes=multimodal_nodes)
        expected = "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        assert str(response) == expected
        assert set(llm.last_called_chat_function) == {"chat", "achat"}
        assert len(llm.last_chat_messages) == 2
        assert [block.block_type for block in llm.last_chat_messages[-1].blocks] == [
            "text",
            "image",
            "text",
        ]
