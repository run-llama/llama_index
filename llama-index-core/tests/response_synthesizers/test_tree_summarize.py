from dataclasses import dataclass
from typing import Any, Sequence

import pytest
from pydantic import BaseModel
from unittest.mock import patch

from llama_index.core import PromptHelper
from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock, MessageRole
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.indices.prompt_helper import DEFAULT_PADDING, ChatPromptHelper
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.chat_prompts import (
    CHAT_TREE_SUMMARIZE_PROMPT,
    CHAT_CONTENT_TREE_SUMMARIZE_PROMPT,
)
from llama_index.core.prompts.default_prompts import DEFAULT_TREE_SUMMARIZE_PROMPT
from llama_index.core.prompts.prompt_utils import (
    get_empty_prompt_messages,
    get_empty_prompt_txt,
)
from llama_index.core.llms.mock import (
    MockLLMWithChatMemoryOfLastCall,
    MockFunctionCallingLLMWithChatMemoryOfLastCall,
)
from llama_index.core.response_synthesizers.tree_summarize import TreeSummarize
from llama_index.core.schema import NodeWithScore, TextNode, ImageNode
from llama_index.core.utilities.token_counting import TokenCounter


class Section(BaseModel):
    heading: str
    text: str


class MockStructuredSummary(BaseModel):
    sections: list[Section]


@dataclass
class SynthVariant:
    llm: MockLLMWithChatMemoryOfLastCall
    multimodal: bool
    prompt_template: BasePromptTemplate
    prompt_helper_cls: type
    helper_kwarg: str
    summary_join: str
    nodes: list[NodeWithScore]


_PNG_1PX_B64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _make_text_variant() -> SynthVariant:
    return SynthVariant(
        llm=MockLLMWithChatMemoryOfLastCall(max_tokens=1),
        multimodal=False,
        prompt_template=DEFAULT_TREE_SUMMARIZE_PROMPT,
        prompt_helper_cls=PromptHelper,
        helper_kwarg="prompt_helper",
        summary_join="\n\n",
        nodes=[
            NodeWithScore(node=TextNode(text="context information1"), score=1.0),
            NodeWithScore(node=TextNode(text="context information2"), score=0.9),
        ],
    )


def _make_text_chat_variant() -> SynthVariant:
    return SynthVariant(
        llm=MockLLMWithChatMemoryOfLastCall(max_tokens=1, is_chat_model=True),
        multimodal=False,
        prompt_template=CHAT_TREE_SUMMARIZE_PROMPT,
        prompt_helper_cls=PromptHelper,
        helper_kwarg="prompt_helper",
        summary_join="\n\n",
        nodes=[
            NodeWithScore(node=TextNode(text="context information1"), score=1.0),
            NodeWithScore(node=TextNode(text="context information2"), score=0.9),
        ],
    )


def _make_multimodal_variant() -> SynthVariant:
    return SynthVariant(
        llm=MockLLMWithChatMemoryOfLastCall(max_tokens=1, is_chat_model=True),
        multimodal=True,
        prompt_template=CHAT_CONTENT_TREE_SUMMARIZE_PROMPT,
        prompt_helper_cls=ChatPromptHelper,
        helper_kwarg="chat_prompt_helper",
        summary_join=" ",
        nodes=[
            NodeWithScore(node=TextNode(text="context information1"), score=1.0),
            NodeWithScore(node=ImageNode(image=_PNG_1PX_B64), score=0.9),
        ],
    )


@pytest.fixture(
    params=[_make_text_variant, _make_text_chat_variant, _make_multimodal_variant],
    ids=["text", "text_chat", "multimodal"],
)
def synth_variant(request: pytest.FixtureRequest) -> SynthVariant:
    return request.param()


@pytest.fixture(
    params=[_make_text_variant, _make_text_chat_variant], ids=["text", "text_chat"]
)
def synth_variant_no_multimodal(request: pytest.FixtureRequest) -> SynthVariant:
    return request.param()


@pytest.fixture(
    params=[_make_text_chat_variant, _make_multimodal_variant],
    ids=["text_chat", "multimodal"],
)
def synth_variant_chat_only(request: pytest.FixtureRequest) -> SynthVariant:
    return request.param()


class TestTreeSummarize:
    def test_init__multimodal_with_non_chat_model_raises_error(self) -> None:
        llm = MockLLMWithChatMemoryOfLastCall(max_tokens=10)
        with pytest.raises(
            ValueError, match="Multimodal synthesis requires a chat LLM."
        ):
            TreeSummarize(llm=llm, multimodal=True)

    def test_synthesize(self, synth_variant: SynthVariant) -> None:
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(llm=llm, multimodal=synth_variant.multimodal)
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        if llm.is_chat_model:
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + 5,
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            multimodal=synth_variant.multimodal,
        )

        response1 = synthesizer1.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function == (
            ["chat"] if llm.is_chat_model else []
        ), "Called once for one compacted chunk. Empty for non chat models"
        if synth_variant.multimodal:
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
        llm.reset_memory()
        response2 = synthesizer2.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function == (
            ["chat"] * 3 if llm.is_chat_model else []
        ), (
            "Two times for compacted chunk split into two "
            "Final time for recursive call on summaries of first two chunks, compacted into a single chunk "
            "Total of 4. "
            "Empty for non chat models"
        )
        assert (
            llm.last_chat_messages is None
            if not llm.is_chat_model
            else synth_variant.summary_join.join(["text", "text"])
            in llm.last_chat_messages[1].content
        ), (
            "The final call was to recursively summarize the summaries from the first two chunks, 'text' and 'text'"
        )
        assert str(response1) == "text"
        assert str(response2) == "text"

    def test_synthesize__use_async(self, synth_variant: SynthVariant) -> None:
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(
            llm=llm, use_async=True, multimodal=synth_variant.multimodal
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        if llm.is_chat_model:
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + 5,
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            use_async=True,
            multimodal=synth_variant.multimodal,
        )

        response1 = synthesizer1.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function == (
            ["chat"] if llm.is_chat_model else []
        ), "Called once for one compacted chunk Doesn't use async for single chunk"
        if synth_variant.multimodal:
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
        llm.reset_memory()
        response2 = synthesizer2.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function.count("achat") == (
            2 if llm.is_chat_model else 0
        ), (
            "Two times async for compacted chunk split into two "
            "Empty for non chat models"
        )
        assert llm.last_called_chat_function.count("chat") == (
            3 if llm.is_chat_model else 0
        ), (
            "Two times for chat being called underhood by by achat"
            "Final time for recursive call on summaries of first two chunks, compacted single chunk doesn't call async "
            "Total of 3. "
            "Since Async calls sync underhood, isolating sync calls with .count"
            "Empty for non chat models"
        )
        assert (
            llm.last_chat_messages is None
            if not llm.is_chat_model
            else synth_variant.summary_join.join(["text", "text"])
            in llm.last_chat_messages[1].content
        ), (
            "The final call was to recursively summarize the summaries from the first two chunks, 'text' and 'text'"
        )
        assert str(response1) == "text"
        assert str(response2) == "text"

    def test_synthesize__streaming(self, synth_variant: SynthVariant) -> None:
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(
            llm=llm, streaming=True, multimodal=synth_variant.multimodal
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        if llm.is_chat_model:
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + 5,
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            streaming=True,
            multimodal=synth_variant.multimodal,
        )

        response1 = synthesizer1.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function == (
            ["stream_chat"] if llm.is_chat_model else []
        ), (
            "Called once for one compacted chunk "
            "Doesn't use async for single chunk "
            "Empty for non chat models"
        )
        if synth_variant.multimodal:
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
        llm.reset_memory()
        response2 = synthesizer2.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function.count("chat") == (
            2 if llm.is_chat_model else 0
        ), (
            "Two times for compacted chunk split into two (not streamed) "
            "Empty for non chat models"
        )
        assert llm.last_called_chat_function.count("stream_chat") == (
            1 if llm.is_chat_model else 0
        ), (
            "Final time for recursive call on summaries of first two chunks, compacted single chunk streamed "
        )
        assert (
            llm.last_chat_messages is None
            if not llm.is_chat_model
            else synth_variant.summary_join.join(["text", "text"])
            in llm.last_chat_messages[1].content
        ), (
            "The final call was to recursively summarize the summaries from the first two chunks, 'text' and 'text'"
        )
        for chunk in response1.response_gen:
            assert chunk == "text"
        for chunk in response2.response_gen:
            assert chunk == "text"

    def test_synthesize__output_cls_default_text_completion_program(
        self, synth_variant_no_multimodal
    ) -> None:
        synth_variant = synth_variant_no_multimodal
        mock_summary_foo = MockStructuredSummary(
            sections=[Section(heading="Foo heading", text="text")]
        )
        mock_summary_bar = MockStructuredSummary(
            sections=[Section(heading="Bar heading", text="text")]
        )
        synth_variant.nodes = [
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.8
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.6
            ),
        ]
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(
            llm=llm,
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        node_tokens = tkn_counter.estimate_tokens_in_messages(
            [
                ChatMessage(
                    blocks=synth_variant.nodes[0].node.get_content_blocks(), role="user"
                ),
            ]
        )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + (node_tokens * 2),
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        with patch.object(
            llm, "_generate_text", return_value=mock_summary_bar.model_dump_json()
        ):
            response1 = synthesizer1.synthesize(query="test", nodes=synth_variant.nodes)
            assert llm.last_called_chat_function == (
                ["chat"] if llm.is_chat_model else []
            ), "Called once for one compacted chunk. Empty for non chat models"
            llm.reset_memory()
            response2 = synthesizer2.synthesize(query="test", nodes=synth_variant.nodes)
            assert len(llm.last_called_chat_function) == (
                3 if llm.is_chat_model else 0
            ), (
                "Two times for compacted chunk split into two. "
                "Final time for recursive call on summaries of first two chunks, compacted single chunk. "
                "Total of 3. "
                "Empty for non chat models"
            )
            assert (
                llm.last_chat_messages is None
                if not llm.is_chat_model
                else synth_variant.summary_join.join(
                    [mock_summary_bar.model_dump_json()] * 2
                )
                in llm.last_chat_messages[1].content
            ), (
                "Final call was to recursively summarize the summaries from the first two chunks"
            )

        assert isinstance(response1, PydanticResponse)
        assert response1.response == mock_summary_bar
        assert isinstance(response2, PydanticResponse)
        assert response2.response == mock_summary_bar

    def test_synthesize__output_cls_default_function_calling_program(
        self, synth_variant_chat_only
    ) -> None:
        synth_variant = synth_variant_chat_only
        mock_summary_foo = MockStructuredSummary(
            sections=[Section(heading="Foo heading", text="text")]
        )
        mock_summary_bar = MockStructuredSummary(
            sections=[Section(heading="Bar heading", text="text")]
        )
        synth_variant.nodes = (
            [
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
                ),
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.8
                ),
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
                ),
                NodeWithScore(
                    node=ImageNode(text=mock_summary_foo.model_dump_json()), score=0.6
                ),
            ]
            if not synth_variant.multimodal
            else [
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
                ),
                NodeWithScore(
                    node=ImageNode(
                        image=(
                            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        ),
                        score=0.8,
                    )
                ),
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
                ),
                NodeWithScore(
                    node=ImageNode(
                        image=(
                            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        ),
                        score=0.6,
                    )
                ),
            ]
        )

        def response_generator(
            _messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatMessage:
            tool_args_json = mock_summary_bar.model_dump_json()
            return ChatMessage(
                blocks=[
                    ToolCallBlock(
                        block_type="tool_call",
                        tool_call_id="call_abc123",
                        tool_name="MockStructuredSummary",
                        tool_kwargs=tool_args_json,
                    )
                ],
                role=MessageRole.ASSISTANT,
            )

        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            max_tokens=1, is_chat_model=True, response_generator=response_generator
        )
        synthesizer1 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=8000
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        if synth_variant.multimodal:
            nodes = synth_variant.nodes
            node_tokens = tkn_counter.estimate_tokens_in_messages(
                [
                    ChatMessage(
                        blocks=nodes[0].node.get_content_blocks()
                        + nodes[1].node.get_content_blocks(),
                        role="user",
                    ),
                ]
            )
        else:
            node_tokens = tkn_counter.estimate_tokens_in_messages(
                [
                    ChatMessage(
                        blocks=synth_variant.nodes[0].node.get_content_blocks() * 2,
                        role="user",
                    ),
                ]
            )
        prompt_helper_kwargs = {
            "context_window": prompt_tokens + DEFAULT_PADDING + node_tokens,
            "num_output": 0,
            "chunk_overlap_ratio": 0,
            "separator": "\n\n",
        }
        if synth_variant.multimodal:
            prompt_helper_kwargs.pop("separator")
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    **prompt_helper_kwargs
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        response1 = synthesizer1.synthesize(query="test", nodes=synth_variant.nodes)
        assert llm.last_called_chat_function == ["chat"], (
            "Called once for one compacted chunk Doesn't use async for single chunk"
        )
        llm.reset_memory()
        response2 = synthesizer2.synthesize(query="test", nodes=synth_variant.nodes)
        assert len(llm.last_called_chat_function) == 3, (
            "Two times for compacted chunk split into two "
            "Final time for recursive call on summaries of first two chunks, compacted single chunk "
            "Total of 3"
        )
        assert (
            synth_variant.summary_join.join([mock_summary_bar.model_dump_json()] * 2)
            in llm.last_chat_messages[1].content
        ), (
            "Final call was to recursively summarize the summaries from the first two chunks"
        )

        assert isinstance(response1, PydanticResponse)
        assert response1.response == mock_summary_bar
        assert isinstance(response2, PydanticResponse)
        assert response2.response == mock_summary_bar

    @pytest.mark.asyncio
    async def test_asynthesize(self, synth_variant: SynthVariant) -> None:
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(llm=llm, multimodal=synth_variant.multimodal)
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        if llm.is_chat_model:
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + 5,
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            multimodal=synth_variant.multimodal,
        )

        response1 = await synthesizer1.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        assert llm.last_called_chat_function.count("achat") == (
            1 if llm.is_chat_model else 0
        ), "Called once for one compacted chunk. Empty for non chat models"
        if synth_variant.multimodal:
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
        llm.reset_memory()
        response2 = await synthesizer2.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        assert llm.last_called_chat_function.count("achat") == (
            3 if llm.is_chat_model else 0
        ), (
            "Two times async for compacted chunk split into two "
            "Final time for recursive call on summaries of first two chunks, compacted into a single chunk "
            "Total of 3. "
            "Empty for non chat models"
        )
        assert llm.last_called_chat_function.count("chat") == (
            3 if llm.is_chat_model else 0
        ), "Achat calls chat under hood. Empty for non chat models"
        assert (
            llm.last_chat_messages is None
            if not llm.is_chat_model
            else synth_variant.summary_join.join(["text", "text"])
            in llm.last_chat_messages[1].content
        ), (
            "The final call was to recursively summarize the summaries from the first two chunks, 'text' and 'text'"
        )
        assert str(response1) == "text"
        assert str(response2) == "text"

    @pytest.mark.asyncio
    async def test_asynthesize__streaming(self, synth_variant: SynthVariant) -> None:
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(
            llm=llm, streaming=True, multimodal=synth_variant.multimodal
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        if llm.is_chat_model:
            prompt_tokens = tkn_counter.estimate_tokens_in_messages(
                get_empty_prompt_messages(qa_template)
            )
        else:
            prompt_tokens = tkn_counter.get_string_tokens(
                get_empty_prompt_txt(qa_template)
            )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + 5,
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            streaming=True,
            multimodal=synth_variant.multimodal,
        )

        response1 = await synthesizer1.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        async for chunk in response1.response_gen:
            assert chunk == "text"
        assert llm.last_called_chat_function.count("astream_chat") == (
            1 if llm.is_chat_model else 0
        ), "Called once for one compacted chunk. Empty for non chat models"
        if synth_variant.multimodal:
            assert [
                block.block_type for block in llm.last_chat_messages[-1].blocks
            ] == ["text", "image", "text"]
        llm.reset_memory()
        response2 = await synthesizer2.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        async for chunk in response2.response_gen:
            assert chunk == "text"
        assert llm.last_called_chat_function.count("achat") == (
            2 if llm.is_chat_model else 0
        ), (
            "Two times async for compacted chunk split into two (not streamed). "
            "Empty for non chat models"
        )
        assert llm.last_called_chat_function.count("astream_chat") == (
            1 if llm.is_chat_model else 0
        ), (
            "Final time for recursive call on summaries of first two chunks, compacted single chunk streamed. "
            "Empty for non chat models"
        )
        assert (
            llm.last_chat_messages is None
            if not llm.is_chat_model
            else synth_variant.summary_join.join(["text", "text"])
            in llm.last_chat_messages[1].content
        ), (
            "The final call was to recursively summarize the summaries from the first two chunks, 'text' and 'text'"
        )

    @pytest.mark.asyncio
    async def test_asynthesize__output_cls_default_text_completion_program(
        self, synth_variant_no_multimodal
    ) -> None:
        synth_variant = synth_variant_no_multimodal
        mock_summary_foo = MockStructuredSummary(
            sections=[Section(heading="Foo heading", text="text")]
        )
        mock_summary_bar = MockStructuredSummary(
            sections=[Section(heading="Bar heading", text="text")]
        )
        synth_variant.nodes = [
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.8
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
            ),
            NodeWithScore(
                node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.6
            ),
        ]
        llm = synth_variant.llm
        synthesizer1 = TreeSummarize(
            llm=llm,
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        node_tokens = tkn_counter.estimate_tokens_in_messages(
            [
                ChatMessage(
                    blocks=synth_variant.nodes[0].node.get_content_blocks(), role="user"
                ),
            ]
        )
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=prompt_tokens + DEFAULT_PADDING + (node_tokens * 2),
                    num_output=0,
                    chunk_overlap_ratio=0,
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        with patch.object(
            llm, "_generate_text", return_value=mock_summary_bar.model_dump_json()
        ):
            response1 = await synthesizer1.asynthesize(
                query="test", nodes=synth_variant.nodes
            )
            assert llm.last_called_chat_function.count("achat") == (
                1 if llm.is_chat_model else 0
            ), "Called once for one compacted chunk. Empty for non chat models"
            llm.reset_memory()
            response2 = await synthesizer2.asynthesize(
                query="test", nodes=synth_variant.nodes
            )
            assert llm.last_called_chat_function.count("achat") == (
                3 if llm.is_chat_model else 0
            ), (
                "Two times for compacted chunk split into two. "
                "Final time for recursive call on summaries of first two chunks, compacted single chunk. "
                "Total of 3. "
                "Empty for non chat models"
            )
            assert llm.last_called_chat_function.count("chat") == (
                3 if llm.is_chat_model else 0
            ), "Achat calls chat under hood. Empty for non chat models"
            assert (
                llm.last_chat_messages is None
                if not llm.is_chat_model
                else synth_variant.summary_join.join(
                    [mock_summary_bar.model_dump_json()] * 2
                )
                in llm.last_chat_messages[1].content
            ), (
                "Final call was to recursively summarize the summaries from the first two chunks"
            )

        assert isinstance(response1, PydanticResponse)
        assert response1.response == mock_summary_bar
        assert isinstance(response2, PydanticResponse)
        assert response2.response == mock_summary_bar

    @pytest.mark.asyncio
    async def test_asynthesize__output_cls_default_function_calling_program(
        self, synth_variant_chat_only
    ) -> None:
        synth_variant = synth_variant_chat_only
        mock_summary_foo = MockStructuredSummary(
            sections=[Section(heading="Foo heading", text="text")]
        )
        mock_summary_bar = MockStructuredSummary(
            sections=[Section(heading="Bar heading", text="text")]
        )
        synth_variant.nodes = (
            [
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
                ),
                NodeWithScore(
                    node=ImageNode(text=mock_summary_foo.model_dump_json()), score=0.8
                ),
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
                ),
                NodeWithScore(
                    node=ImageNode(text=mock_summary_foo.model_dump_json()), score=0.6
                ),
            ]
            if not synth_variant.multimodal
            else [
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.9
                ),
                NodeWithScore(
                    node=ImageNode(
                        image=(
                            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        ),
                        score=0.8,
                    )
                ),
                NodeWithScore(
                    node=TextNode(text=mock_summary_foo.model_dump_json()), score=0.7
                ),
                NodeWithScore(
                    node=ImageNode(
                        image=(
                            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        ),
                        score=0.6,
                    )
                ),
            ]
        )

        def response_generator(
            _messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatMessage:
            tool_args_json = mock_summary_bar.model_dump_json()
            return ChatMessage(
                blocks=[
                    ToolCallBlock(
                        block_type="tool_call",
                        tool_call_id="call_abc123",
                        tool_name="MockStructuredSummary",
                        tool_kwargs=tool_args_json,
                    )
                ],
                role=MessageRole.ASSISTANT,
            )

        llm = MockFunctionCallingLLMWithChatMemoryOfLastCall(
            max_tokens=1, is_chat_model=True, response_generator=response_generator
        )
        synthesizer1 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    context_window=8000
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )
        tkn_counter = TokenCounter()
        qa_template = synth_variant.prompt_template.partial_format(query_str="test")
        prompt_tokens = tkn_counter.estimate_tokens_in_messages(
            get_empty_prompt_messages(qa_template)
        )
        if synth_variant.multimodal:
            nodes = synth_variant.nodes
            node_tokens = tkn_counter.estimate_tokens_in_messages(
                [
                    ChatMessage(
                        blocks=nodes[0].node.get_content_blocks()
                        + nodes[1].node.get_content_blocks(),
                        role="user",
                    ),
                ]
            )
        else:
            node_tokens = tkn_counter.estimate_tokens_in_messages(
                [
                    ChatMessage(
                        blocks=synth_variant.nodes[0].node.get_content_blocks() * 2,
                        role="user",
                    ),
                ]
            )
        prompt_helper_kwargs = {
            "context_window": prompt_tokens + DEFAULT_PADDING + node_tokens,
            "num_output": 0,
            "chunk_overlap_ratio": 0,
            "separator": "\n\n",
        }
        if synth_variant.multimodal:
            prompt_helper_kwargs.pop("separator")
        synthesizer2 = TreeSummarize(
            llm=llm,
            **{
                synth_variant.helper_kwarg: synth_variant.prompt_helper_cls(
                    **prompt_helper_kwargs
                )
            },
            output_cls=MockStructuredSummary,
            multimodal=synth_variant.multimodal,
        )

        response1 = await synthesizer1.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        assert llm.last_called_chat_function.count("achat") == 1, (
            "Called once for one compacted chunk. "
        )
        llm.reset_memory()
        response2 = await synthesizer2.asynthesize(
            query="test", nodes=synth_variant.nodes
        )
        assert llm.last_called_chat_function.count("achat") == 3, (
            "Two times async for compacted chunk split into two. "
            "Final time for recursive call on summaries of first two chunks, compacted single chunk. "
            "Total of 3"
        )
        assert (
            synth_variant.summary_join.join([mock_summary_bar.model_dump_json()] * 2)
            in llm.last_chat_messages[1].content
        ), (
            "Final call was to recursively summarize the summaries from the first two chunks"
        )

        assert isinstance(response1, PydanticResponse)
        assert response1.response == mock_summary_bar
        assert isinstance(response2, PydanticResponse)
        assert response2.response == mock_summary_bar
