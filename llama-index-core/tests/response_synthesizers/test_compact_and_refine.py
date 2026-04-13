import pytest
from unittest.mock import patch

from llama_index.core import PromptHelper
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.indices.prompt_helper import ChatPromptHelper
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.prompts.chat_prompts import (
    CHAT_CONTENT_QA_PROMPT,
    CHAT_CONTENT_REFINE_PROMPT,
)
from llama_index.core.prompts.prompt_utils import (
    get_biggest_prompt,
    get_empty_prompt_messages,
)
from llama_index.core.response_synthesizers.refine import Refine, MultimodalRefine
from llama_index.core.response_synthesizers.compact_and_refine import (
    CompactAndRefine,
    MultimodalCompactAndRefine,
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


class TestCompactAndRefine:
    def test_synthesize(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndRefine(llm=llm)
        tkn_counter = TokenCounter()
        max_prompt = get_biggest_prompt(
            [
                prompt.partial_format(query_str="test")
                for prompt in list(synthesizer1.get_prompts().values())
            ]
        )
        max_prompt = max_prompt.select(llm=llm)
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = CompactAndRefine(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
        )
        with patch.object(
            Refine, "get_response", wraps=Refine(llm=llm).get_response
        ) as wraps_refine_get_response:
            response1 = synthesizer1.synthesize(query="test", nodes=nodes)
            response2 = synthesizer2.synthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == ["chat"] * 4, (
            "First call one compacted node into one = 1 call "
            "Second call compacted node split into 3 = 3 calls "
            "Total of 4 calls"
        )
        assert wraps_refine_get_response.call_args_list[0].kwargs["text_chunks"] == [
            "context information1\n\ncontext information2"
        ]
        assert wraps_refine_get_response.call_args_list[1].kwargs["text_chunks"] == [
            "context information1",
            "context",
            "information2",
        ]

    def test_synthesize__streaming(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndRefine(llm=llm, streaming=True)
        tkn_counter = TokenCounter()
        max_prompt = get_biggest_prompt(
            [
                prompt.partial_format(query_str="test")
                for prompt in list(synthesizer1.get_prompts().values())
            ]
        )
        max_prompt = max_prompt.select(llm=llm)
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = CompactAndRefine(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
            streaming=True,
        )
        with patch.object(
            Refine, "get_response", wraps=Refine(llm=llm, streaming=True).get_response
        ) as wraps_refine_get_response:
            response1 = synthesizer1.synthesize(query="test", nodes=nodes)
            response2 = synthesizer2.synthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"stream_chat"}
        assert len(llm.last_called_chat_function) == 4, (
            "First call one compacted node into one = 1 call "
            "Second call compacted node split into 3 = 3 calls "
            "Total of 4 calls"
        )
        assert wraps_refine_get_response.call_args_list[0].kwargs["text_chunks"] == [
            "context information1\n\ncontext information2"
        ]
        assert wraps_refine_get_response.call_args_list[1].kwargs["text_chunks"] == [
            "context information1",
            "context",
            "information2",
        ]

    @pytest.mark.asyncio
    async def test_asynthesize(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndRefine(llm=llm)
        tkn_counter = TokenCounter()
        max_prompt = get_biggest_prompt(
            [
                prompt.partial_format(query_str="test")
                for prompt in list(synthesizer1.get_prompts().values())
            ]
        )
        max_prompt = max_prompt.select(llm=llm)
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = CompactAndRefine(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
        )
        with patch.object(
            Refine, "aget_response", wraps=Refine(llm=llm).aget_response
        ) as wraps_refine_aget_response:
            response1 = await synthesizer1.asynthesize(query="test", nodes=nodes)
            response2 = await synthesizer2.asynthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"achat", "chat"}
        assert len(llm.last_called_chat_function) == 4 * 2, (
            "First call one compacted node = 1 call "
            "Second call compacted node split into 3 = 3 calls "
            "For Async, achat and chat are called (multiplied by 2)"
            "Total of 8 calls"
        )
        assert wraps_refine_aget_response.call_args_list[0].kwargs["text_chunks"] == [
            "context information1\n\ncontext information2"
        ]
        assert wraps_refine_aget_response.call_args_list[1].kwargs["text_chunks"] == [
            "context information1",
            "context",
            "information2",
        ]

    @pytest.mark.asyncio
    async def test_asynthesize__streaming(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = CompactAndRefine(llm=llm, streaming=True)
        tkn_counter = TokenCounter()
        max_prompt = get_biggest_prompt(
            [
                prompt.partial_format(query_str="test")
                for prompt in list(synthesizer1.get_prompts().values())
            ]
        )
        max_prompt = max_prompt.select(llm=llm)
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = CompactAndRefine(
            llm=llm,
            prompt_helper=PromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
            streaming=True,
        )
        with patch.object(
            Refine, "aget_response", wraps=Refine(llm=llm, streaming=True).aget_response
        ) as wraps_refine_aget_response:
            response1 = await synthesizer1.asynthesize(query="test", nodes=nodes)
            response2 = await synthesizer2.asynthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"astream_chat", "stream_chat"}
        assert len(llm.last_called_chat_function) == 4 * 2, (
            "First call one compacted node = 1 call "
            "Second call compacted node split into 3 = 3 calls "
            "For Async, achat and chat are called (multiplied by 2)"
            "Total of 8 calls"
        )
        assert wraps_refine_aget_response.call_args_list[0].kwargs["text_chunks"] == [
            "context information1\n\ncontext information2"
        ]
        assert wraps_refine_aget_response.call_args_list[1].kwargs["text_chunks"] == [
            "context information1",
            "context",
            "information2",
        ]


class TestMultimodalCompactAndRefine:
    def test_synthesize(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = MultimodalCompactAndRefine(llm=llm)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        refine_template = CHAT_CONTENT_REFINE_PROMPT.partial_format(query_str="test")
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = MultimodalCompactAndRefine(
            llm=llm,
            prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
        )
        with patch.object(
            MultimodalRefine,
            "get_response",
            wraps=MultimodalRefine(llm=llm).get_response,
        ) as wraps_get_response:
            response1 = synthesizer1.synthesize(query="test", nodes=nodes)
            response2 = synthesizer2.synthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert llm.last_called_chat_function == ["chat"] * 3, (
            "First call one compacted node = 1 call "
            "Second call one compacted node split into 2 = 2 calls "
            "Total of 3 calls"
        )
        assert wraps_get_response.call_args_list[0].kwargs["message_chunks"] == [
            ChatMessage(content="context information1 context information2")
        ]
        assert wraps_get_response.call_args_list[1].kwargs["message_chunks"] == [
            ChatMessage(content="context information1"),
            ChatMessage(content="context information2"),
        ]

    def test_synthesize__streaming(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = MultimodalCompactAndRefine(llm=llm, streaming=True)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        refine_template = CHAT_CONTENT_REFINE_PROMPT.partial_format(query_str="test")
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = MultimodalCompactAndRefine(
            llm=llm,
            prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
            streaming=True,
        )
        with patch.object(
            MultimodalRefine,
            "get_response",
            wraps=MultimodalRefine(llm=llm, streaming=True).get_response,
        ) as wraps_get_response:
            response1 = synthesizer1.synthesize(query="test", nodes=nodes)
            response2 = synthesizer2.synthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"stream_chat"}
        assert len(llm.last_called_chat_function) == 3, (
            "First call one compacted node = 1 call "
            "Second call one compacted node split into 2 = 2 calls "
            "Total of 3 calls"
        )
        assert wraps_get_response.call_args_list[0].kwargs["message_chunks"] == [
            ChatMessage(content="context information1 context information2")
        ]
        assert wraps_get_response.call_args_list[1].kwargs["message_chunks"] == [
            ChatMessage(content="context information1"),
            ChatMessage(content="context information2"),
        ]

    @pytest.mark.asyncio
    async def test_asynthesize(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = MultimodalCompactAndRefine(llm=llm)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        refine_template = CHAT_CONTENT_REFINE_PROMPT.partial_format(query_str="test")
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = MultimodalCompactAndRefine(
            llm=llm,
            prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
        )
        with patch.object(
            MultimodalRefine,
            "aget_response",
            wraps=MultimodalRefine(llm=llm).aget_response,
        ) as wraps_aget_response:
            response1 = await synthesizer1.asynthesize(query="test", nodes=nodes)
            response2 = await synthesizer2.asynthesize(query="test", nodes=nodes)
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"achat", "chat"}
        assert len(llm.last_called_chat_function) == 3 * 2, (
            "First call one compacted node = 1 call "
            "Second call one compacted node split into 2 = 2 calls "
            "For Async, astream_chat and stream_chat are called (multiplied by 2) "
            "Total of 6 calls"
        )
        assert wraps_aget_response.call_args_list[0].kwargs["message_chunks"] == [
            ChatMessage(content="context information1 context information2")
        ]
        assert wraps_aget_response.call_args_list[1].kwargs["message_chunks"] == [
            ChatMessage(content="context information1"),
            ChatMessage(content="context information2"),
        ]

    @pytest.mark.asyncio
    async def test_asynthesize__streaming(self, nodes: list[NodeWithScore]) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10, is_chat_model=True)
        synthesizer1 = MultimodalCompactAndRefine(llm=llm, streaming=True)
        tkn_counter = TokenCounter()
        qa_template = CHAT_CONTENT_QA_PROMPT.partial_format(query_str="test")
        refine_template = CHAT_CONTENT_REFINE_PROMPT.partial_format(query_str="test")
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(max_prompt)
        )
        synthesizer2 = MultimodalCompactAndRefine(
            llm=llm,
            prompt_helper=ChatPromptHelper(
                context_window=prompt_tokens + 3, num_output=0, chunk_overlap_ratio=0
            ),
            response_padding_size=0,
            streaming=True,
        )
        with patch.object(
            MultimodalRefine,
            "aget_response",
            wraps=MultimodalRefine(llm=llm, streaming=True).aget_response,
        ) as wraps_aget_response:
            response1 = await synthesizer1.asynthesize(query="test", nodes=nodes)
            response2 = await synthesizer2.asynthesize(query="test", nodes=nodes)

        # Assert
        assert str(response1) == " ".join(["text"] * 10)
        assert str(response2) == " ".join(["text"] * 10)
        assert set(llm.last_called_chat_function) == {"astream_chat", "stream_chat"}
        assert len(llm.last_called_chat_function) == 3 * 2, (
            "First call one compacted node = 1 call "
            "Second call one compacted node split into 2 = 2 calls "
            "For Async, astream_chat and stream_chat are called (multiplied by 2) "
            "Total of 6 calls"
        )
        assert wraps_aget_response.call_args_list[0].kwargs["message_chunks"] == [
            ChatMessage(content="context information1 context information2")
        ]
        assert wraps_aget_response.call_args_list[1].kwargs["message_chunks"] == [
            ChatMessage(content="context information1"),
            ChatMessage(content="context information2"),
        ]
