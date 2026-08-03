import pytest
from unittest.mock import patch

from llama_index.core import PromptHelper
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock
from llama_index.core.indices.prompt_helper import DEFAULT_PADDING, ChatPromptHelper
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.prompts.chat_prompts import (
    CHAT_CONTENT_QA_PROMPT,
    CHAT_TEXT_QA_PROMPT,
)
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.prompts.prompt_utils import (
    get_empty_prompt_messages,
    get_empty_prompt_txt,
)
from llama_index.core.response_synthesizers.accumulate import Accumulate
from llama_index.core.response_synthesizers.compact_and_accumulate import (
    CompactAndAccumulate,
)
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode
from llama_index.core.utilities.token_counting import TokenCounter


@pytest.fixture()
def nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="context information1"), score=1.0),
        NodeWithScore(node=TextNode(text="context information2"), score=0.9),
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


class TestCompactAndAccumulate:
    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    def test_synthesize(self, nodes: list[NodeWithScore], llm) -> None:
        synthesizer1 = CompactAndAccumulate(llm=llm)
        tkn_counter = TokenCounter()
        if llm.is_chat_model:
            qa_template = CHAT_TEXT_QA_PROMPT.partial_format(query_str="test")
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            qa_template = DEFAULT_TEXT_QA_PROMPT.partial_format(query_str="test")
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = CompactAndAccumulate(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + DEFAULT_PADDING + 3,
                num_output=0,
                chunk_overlap_ratio=0,
            ),
        )
        with patch.object(
            Accumulate, "get_response", wraps=Accumulate(llm=llm).get_response
        ) as wraps_accumulate_get_response:
            response1 = synthesizer1.synthesize(query="test", nodes=nodes)
            assert llm.last_called_chat_function == (
                ["chat"] if llm.is_chat_model else []
            ), "First call one compacted node = 1 call Empty for non chat models"
            assert (
                llm.last_chat_messages is None
                if not llm.is_chat_model
                else len(llm.last_chat_messages) == 2
            ), "System and User messagesEmpty for non chat models"
            llm.reset_memory()
            response2 = synthesizer2.synthesize(query="test", nodes=nodes)
            assert llm.last_called_chat_function == (
                ["chat"] * 3 if llm.is_chat_model else []
            ), (
                "Second call compacted node split into 3 = 3 calls "
                "Empty for non chat models"
            )
            assert (
                llm.last_chat_messages is None
                if not llm.is_chat_model
                else len(llm.last_chat_messages) == 2
            ), "System and User messagesEmpty for non chat models"
        assert str(response1) == f"Response 1: {' '.join(['text'] * 10)}"
        assert str(response2) == "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(3)]
        )

        assert wraps_accumulate_get_response.call_args_list[0].kwargs[
            "text_chunks"
        ] == ["context information1\n\ncontext information2"]
        assert wraps_accumulate_get_response.call_args_list[1].kwargs[
            "text_chunks"
        ] == ["context information1", "context", "information2"]

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    def test_synthesize__streaming_raises_error(
        self, nodes: list[NodeWithScore], llm
    ) -> None:
        synthesizer = CompactAndAccumulate(llm=llm, streaming=True)
        with pytest.raises(ValueError, match="Unable to stream"):
            synthesizer.synthesize(query="test", nodes=nodes)

    def test_synthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore], png_1px_b64
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndAccumulate(llm=llm, multimodal=True)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        synthesizer2 = CompactAndAccumulate(
            llm=llm,
            chat_prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + DEFAULT_PADDING + 3,
                num_output=0,
                chunk_overlap_ratio=0,
            ),
            multimodal=True,
        )
        with patch.object(
            Accumulate,
            "get_response_from_messages",
            wraps=Accumulate(llm=llm, multimodal=True).get_response_from_messages,
        ) as wraps_accumulate_get_response_from_messages:
            response1 = synthesizer1.synthesize(query="test", nodes=multimodal_nodes)
            assert llm.last_called_chat_function == ["chat"]
            assert len(llm.last_chat_messages) == 2, "System and User messages"
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
            llm.reset_memory()
            response2 = synthesizer2.synthesize(query="test", nodes=multimodal_nodes)
            assert llm.last_called_chat_function == ["chat"] * 2, (
                "Compacted node split into 2 = 2 calls "
            )
            assert len(llm.last_chat_messages) == 2, "System and User messages"
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]

        assert str(response1) == f"Response 1: {' '.join(['text'] * 10)}"
        assert str(response2) == "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        assert wraps_accumulate_get_response_from_messages.call_args_list[0].kwargs[
            "message_chunks"
        ] == [
            ChatMessage(
                content=[
                    TextBlock(text="context information"),
                    ImageBlock(image=png_1px_b64),
                ]
            )
        ]
        assert wraps_accumulate_get_response_from_messages.call_args_list[1].kwargs[
            "message_chunks"
        ] == [
            ChatMessage(content=[TextBlock(text="context information")]),
            ChatMessage(content=[ImageBlock(image=png_1px_b64)]),
        ]

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize(self, nodes: list[NodeWithScore], llm) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndAccumulate(llm=llm)
        tkn_counter = TokenCounter()
        if llm.is_chat_model:
            qa_template = CHAT_TEXT_QA_PROMPT.partial_format(query_str="test")
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            qa_template = DEFAULT_TEXT_QA_PROMPT.partial_format(query_str="test")
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = CompactAndAccumulate(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + DEFAULT_PADDING + 3,
                num_output=0,
                chunk_overlap_ratio=0,
            ),
        )
        with patch.object(
            Accumulate, "aget_response", wraps=Accumulate(llm=llm).aget_response
        ) as wraps_accumulate_aget_response:
            response1 = await synthesizer1.asynthesize(query="test", nodes=nodes)
            response2 = await synthesizer2.asynthesize(query="test", nodes=nodes)
        assert str(response1) == f"Response 1: {' '.join(['text'] * 10)}"
        assert str(response2) == "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(3)]
        )
        assert set(llm.last_called_chat_function) == (
            {"achat", "chat"} if llm.is_chat_model else {}
        ), "Achat calls chat under hood. Empty for non chat models"
        assert len(llm.last_called_chat_function) == (8 if llm.is_chat_model else 0), (
            "First call one compacted node = 1 call "
            "Second call compacted node split into 3 = 3 calls "
            "Achat and chat every time, multiply by 2"
            "Total of 8 calls. "
            "Empty for non chat models"
        )
        assert wraps_accumulate_aget_response.call_args_list[0].kwargs[
            "text_chunks"
        ] == ["context information1\n\ncontext information2"]
        assert wraps_accumulate_aget_response.call_args_list[1].kwargs[
            "text_chunks"
        ] == ["context information1", "context", "information2"]

    @pytest.mark.parametrize(
        "llm",
        [
            MockLLMWithChatMemoryOfLastCall(max_tokens=10),
            MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True),
        ],
    )
    @pytest.mark.asyncio
    async def test_asynthesize__streaming_raises_error(
        self, nodes: list[NodeWithScore], llm
    ) -> None:
        synthesizer = CompactAndAccumulate(llm=llm, streaming=True)
        with pytest.raises(ValueError, match="Unable to stream"):
            await synthesizer.asynthesize(query="test", nodes=nodes)

    @pytest.mark.asyncio
    async def test_asynthesize__multimodal(
        self, multimodal_nodes: list[NodeWithScore], png_1px_b64
    ) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndAccumulate(llm=llm, multimodal=True)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        synthesizer2 = CompactAndAccumulate(
            llm=llm,
            chat_prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + DEFAULT_PADDING + 3,
                num_output=0,
                chunk_overlap_ratio=0,
            ),
            multimodal=True,
        )
        with patch.object(
            Accumulate,
            "aget_response_from_messages",
            wraps=Accumulate(llm=llm, multimodal=True).aget_response_from_messages,
        ) as wraps_accumulate_aget_response_from_messages:
            response1 = await synthesizer1.asynthesize(
                query="test", nodes=multimodal_nodes
            )
            assert llm.last_called_chat_function.count("achat") == 1, (
                "Since async calls sync under hood, we are isolating async calls here with .count"
            )
            assert len(llm.last_chat_messages) == 2, "System and User messages"
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
            llm.reset_memory()
            response2 = await synthesizer2.asynthesize(
                query="test", nodes=multimodal_nodes
            )
            assert llm.last_called_chat_function.count("achat") == 2, (
                "Compacted node split into 2 = 2 calls. "
                "Since async calls sync under hood, we are isolating async calls here with .count"
            )
            assert len(llm.last_chat_messages) == 2, "System and User messages"
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]

        assert str(response1) == f"Response 1: {' '.join(['text'] * 10)}"
        assert str(response2) == "\n---------------------\n".join(
            [f"Response {i + 1}: {' '.join(['text'] * 10)}" for i in range(2)]
        )
        assert wraps_accumulate_aget_response_from_messages.call_args_list[0].kwargs[
            "message_chunks"
        ] == [
            ChatMessage(
                content=[
                    TextBlock(text="context information"),
                    ImageBlock(image=png_1px_b64),
                ]
            )
        ]
        assert wraps_accumulate_aget_response_from_messages.call_args_list[1].kwargs[
            "message_chunks"
        ] == [
            ChatMessage(content=[TextBlock(text="context information")]),
            ChatMessage(content=[ImageBlock(image=png_1px_b64)]),
        ]
