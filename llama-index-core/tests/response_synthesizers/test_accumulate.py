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

    @pytest.mark.asyncio
    async def test_direct_get_response_and_from_messages_across_multimodal_flag(
        self,
    ) -> None:
        from llama_index.core.base.llms.types import ChatMessage, TextBlock

        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        msg_chunks = [ChatMessage(blocks=[TextBlock(text="msg chunk")])]
        txt_chunks = ["text chunk"]

        # Default (multimodal=False) should still handle get_response_from_messages
        synth_default = Accumulate(llm=llm, multimodal=False)
        assert synth_default._make_prompt_kwargs("chunk") == {"context_str": "chunk"}
        assert synth_default._make_prompt_kwargs(msg_chunks[0]) == {
            "context_messages": [msg_chunks[0]]
        }
        res_sync = synth_default.get_response_from_messages("test", msg_chunks)
        res_async = await synth_default.aget_response_from_messages("test", msg_chunks)
        assert str(res_sync).startswith("Response 1: ")
        assert str(res_async).startswith("Response 1: ")

        # multimodal=True should still handle get_response with string chunks
        synth_mm = Accumulate(llm=llm, multimodal=True)
        res_txt_sync = synth_mm.get_response("test", txt_chunks)
        res_txt_async = await synth_mm.aget_response("test", txt_chunks)
        assert str(res_txt_sync).startswith("Response 1: ")
        assert str(res_txt_async).startswith("Response 1: ")

        # Empty chunks should return "Empty Response"
        assert synth_default.get_response("test", []) == "Empty Response"
        assert synth_default.get_response_from_messages("test", []) == "Empty Response"
