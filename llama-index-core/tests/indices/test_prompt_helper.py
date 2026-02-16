"""Test PromptHelper."""

import base64
import httpx
from typing import Optional, Type, Union

import pytest

from llama_index.core.base.llms.types import (
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    DocumentBlock,
)
from llama_index.core.indices.prompt_helper import ChatPromptHelper, PromptHelper
from llama_index.core.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser.text.utils import truncate_text
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_utils import (
    get_biggest_prompt,
    get_biggest_chat_prompt,
    get_empty_prompt_txt,
    get_empty_prompt_messages,
)
from llama_index.core.schema import TextNode
from tests.mock_utils.mock_utils import mock_tokenizer


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.fixture()
def mock_pdf_bytes(pdf_url) -> bytes:
    """
    Returns a byte string representing a very simple, minimal PDF file.
    """
    return httpx.get(pdf_url).content


@pytest.fixture()
def pdf_base64(mock_pdf_bytes) -> bytes:
    return base64.b64encode(mock_pdf_bytes)


@pytest.mark.parametrize(
    ("prompt", "chunk_size_limit", "num_chunks", "padding", "expected"),
    [
        pytest.param("This is the prompt", None, 1, 6, 0, id="one_chunk"),
        pytest.param("This is the prompt", None, 2, 3, 0, id="two_chunks_no_limit"),
        pytest.param("This is the prompt", 2, 2, 0, 2, id="two_chunks_with_limit"),
        pytest.param("This is the prompt", None, 2, 2, 1, id="two_chunks_with_padding"),
        pytest.param(
            (
                "A really really really really really really really really"
                " really really really really long prompt"
            ),
            None,
            2,
            0,
            ValueError,
            id="misconfigured_chunks_denied",
        ),
    ],
)
def test_prompt_helper_get_chunk_size(
    prompt: str,
    chunk_size_limit: Optional[int],
    num_chunks: int,
    padding: int,
    expected: Union[int, Type[Exception]],
) -> None:
    """Test get chunk size given prompt."""
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        chunk_size_limit=chunk_size_limit,
    )
    if isinstance(expected, int):
        chunk_size = prompt_helper._get_available_chunk_size(
            PromptTemplate(prompt), num_chunks, padding=padding
        )
        assert chunk_size == expected
    else:
        with pytest.raises(expected):
            prompt_helper._get_available_chunk_size(
                PromptTemplate(prompt), num_chunks, padding=padding
            )


def test_prompt_helper_get_text_splitter() -> None:
    """Test get text splitter."""
    test_prompt_text = "This is the prompt{text}"
    test_prompt = PromptTemplate(test_prompt_text)
    prompt_helper = PromptHelper(
        context_window=11, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    assert text_splitter.chunk_size == 2
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = truncate_text(test_text, text_splitter)
    assert truncated_text == "Hello world"

    # test with chunk_size_limit
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        chunk_size_limit=1,
    )
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello", "world", "foo", "Hello", "world", "bar"]


def test_prompt_helper_get_text_splitter_partial() -> None:
    """Test get text splitter with a partially formatted prompt."""
    # test without partially formatting
    test_prompt_text = "This is the {foo} prompt{text}"
    test_prompt = PromptTemplate(test_prompt_text)
    prompt_helper = PromptHelper(
        context_window=11, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = truncate_text(test_text, text_splitter)
    assert truncated_text == "Hello world"

    # test with partially formatting
    test_prompt = PromptTemplate(test_prompt_text)
    test_prompt = test_prompt.partial_format(foo="bar")
    prompt_helper = PromptHelper(
        context_window=12, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    assert get_empty_prompt_txt(test_prompt) == "This is the bar prompt"
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = truncate_text(test_text, text_splitter)
    assert truncated_text == "Hello world"


def test_prompt_helper_truncate() -> None:
    """Test truncate."""
    # test prompt uses up one token
    test_prompt_txt = "test{text}"
    test_prompt = PromptTemplate(test_prompt_txt)
    # set context_window=19
    # For each text chunk, there's 4 tokens for text + 5 for the padding
    prompt_helper = PromptHelper(
        context_window=19, num_output=0, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    text_chunks = ["This is a test foo bar", "Hello world bar foo"]

    truncated_chunks = prompt_helper.truncate(
        prompt=test_prompt, text_chunks=text_chunks
    )
    assert truncated_chunks == [
        "This is a test",
        "Hello world bar foo",
    ]


def test_prompt_helper_get_numbered_text_from_nodes() -> None:
    """Test get_text_from_nodes."""
    # test prompt uses up one token
    test_prompt_txt = "test{text}"
    test_prompt = PromptTemplate(test_prompt_txt)
    # set context_window=17
    # For each text chunk, there's 3 for text, 5 for padding (including number)
    prompt_helper = PromptHelper(
        context_window=17, num_output=0, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    node1 = TextNode(text="This is a test foo bar")
    node2 = TextNode(text="Hello world bar foo")

    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        prompt=test_prompt,
        num_chunks=2,
    )
    response = get_numbered_text_from_nodes([node1, node2], text_splitter=text_splitter)
    assert str(response) == ("(1) This is a\n\n(2) Hello world bar")


def test_prompt_helper_repack() -> None:
    """Test repack."""
    test_prompt_text = "This is the prompt{text}"
    test_prompt = PromptTemplate(test_prompt_text)
    prompt_helper = PromptHelper(
        context_window=13,
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
    )
    text_chunks = ["Hello", "world", "foo", "Hello", "world", "bar"]
    compacted_chunks = prompt_helper.repack(test_prompt, text_chunks)
    assert compacted_chunks == ["Hello\n\nworld\n\nfoo", "Hello\n\nworld\n\nbar"]


def test_prompt_helper_get_biggest_prompt() -> None:
    """Test get_biggest_prompt from PromptHelper."""
    prompt1 = PromptTemplate("This is the prompt{text}")
    prompt2 = PromptTemplate("This is the longer prompt{text}")
    prompt3 = PromptTemplate("This is the {text}")
    biggest_prompt = get_biggest_prompt([prompt1, prompt2, prompt3])

    assert biggest_prompt == prompt2


def test_prompt_helper_json_in_prompt() -> None:
    """Test that a JSON object in the prompt doesn't cause an error."""
    # test with normal prompt
    prompt = PromptTemplate(
        'This is the prompt {text} but it also has {"json": "in it"}'
    )
    prompt.partial_format(text="hello_world")
    prompt_helper = PromptHelper()

    texts = prompt_helper.repack(prompt, ["hello_world"])
    assert len(texts) == 1

    # test with chat messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system",
                content="You are a helpful assistant that knows about {topic}. Here's some JSON: {'foo': 'bar'}",
            ),
            ChatMessage(
                role="user",
                content="What is the capital of the moon? Here's some JSON: {'foo': 'bar'}",
            ),
        ]
    )
    prompt.partial_format(topic="the moon")
    prompt_helper = PromptHelper()

    texts = prompt_helper.repack(prompt, ["hello_world"])
    assert len(texts) == 1

    # test with more complex JSON
    prompt = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful assistant that knows about {topic}\n"
                    "Here's some JSON: {'foo': 'bar'}\n"
                    "here's some other stuff: {key: val for val in d.items()}\n"
                ),
            ),
            ChatMessage(
                role="user",
                content="What is the capital of the moon? Here's some JSON: {'foo': 'bar'}",
            ),
        ]
    )
    prompt.partial_format(topic="the moon")
    prompt_helper = PromptHelper()

    texts = prompt_helper.repack(prompt, ["hello_world"])
    assert len(texts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "chunk_size_limit", "num_chunks", "padding", "expected"),
    [
        pytest.param("This is the prompt", None, 1, 6, 0, id="one_chunk"),
        pytest.param("This is the prompt", None, 2, 3, 0, id="two_chunks_no_limit"),
        pytest.param("This is the prompt", 2, 2, 0, 2, id="two_chunks_with_limit"),
        pytest.param("This is the prompt", None, 2, 2, 1, id="two_chunks_with_padding"),
        pytest.param(
            (
                "A really really really really really really really really"
                " really really really really long prompt"
            ),
            None,
            2,
            0,
            ValueError,
            id="misconfigured_chunks_denied",
        ),
    ],
)
async def test_chat_prompt_helper_aget_available_chunk_size_text_prompts(
    prompt: str,
    chunk_size_limit: Optional[int],
    num_chunks: int,
    padding: int,
    expected: Union[int, Type[Exception]],
) -> None:
    """Test get chunk size given prompt."""
    prompt_helper = ChatPromptHelper(
        context_window=15,  # Adjusted for chat message overhead 1 per role and 3 per message
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        chunk_size_limit=chunk_size_limit,
    )
    if isinstance(expected, int):
        chunk_size = await prompt_helper._aget_available_chunk_size(
            ChatPromptTemplate(message_templates=[ChatMessage(content=prompt)]),
            num_chunks,
            padding=padding,
        )
        assert chunk_size == expected
    else:
        with pytest.raises(expected):
            await prompt_helper._aget_available_chunk_size(
                ChatPromptTemplate(
                    message_templates=[ChatMessage(blocks=[TextBlock(text=prompt)])]
                ),
                num_chunks,
                padding=padding,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "non_text_block",
    [
        ImageBlock(image=b"{image_bytes}"),
        AudioBlock(audio=b"{audio_bytes}"),
        VideoBlock(video=b"{video_bytes}"),
        DocumentBlock(data=b"{pdf_bytes}"),
    ],
)
@pytest.mark.parametrize("num_chunks", [1, 5, 10, 25])
@pytest.mark.parametrize("context_window", [4096, 8192])
@pytest.mark.parametrize("num_output", [256, 512])
@pytest.mark.parametrize("padding", [0, 5])
async def test_chat_prompt_helper_aget_available_chunk_size_counts_non_text_blocks(
    png_1px,
    non_text_block,
    num_chunks,
    context_window,
    num_output,
    padding,
) -> None:
    """
    Test that non-text blocks are counted correctly.

    chunk_size = (context_window - num_prompt_tokens - self.num_output) // num_chunks - padding

    In this test, the difference in num_prompt_tokens when including a non-text block should be equal to calling
    non_text_block.estimate_tokens().
    """
    kwargs = {
        "image_bytes": png_1px,
        # When ffmpeg is not installed, audio and video blocks have fixed estimates, so the actual bytes don't matter.
        "audio_bytes": b"fake_audio",
        "video_bytes": b"fake_video",
        "pdf_bytes": b"fake_pdf",
    }

    relevant_kwargs = {
        k: v for k, v in kwargs.items() if k in non_text_block.get_template_vars()
    }
    nt_block = non_text_block.format_vars(**relevant_kwargs)
    text = "Tell me a joke that is thematic to this image/audio/video."
    prompt_wo_non_text = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system", blocks=[TextBlock(text="You're a a helpful assistant.")]
            ),
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text=text),
                ],
            ),
        ]
    )

    prompt_w_non_text = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system", blocks=[TextBlock(text="You're a a helpful assistant.")]
            ),
            ChatMessage(role="user", blocks=[TextBlock(text=text), nt_block]),
        ]
    )

    prompt_helper = ChatPromptHelper(
        context_window=context_window,
        num_output=num_output,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # Chunk size limit set trivially to context window size so algebra is simpler
        chunk_size_limit=context_window,
    )

    # Actual
    chunk_size_w_non_text = await prompt_helper._aget_available_chunk_size(
        prompt_w_non_text, num_chunks=num_chunks, padding=padding
    )

    # Expected
    num_prompt_tokens_wo_non_text = (
        await prompt_helper._token_counter.aestimate_tokens_in_messages(
            get_empty_prompt_messages(prompt_wo_non_text)
        )
    )
    expected_num_prompt_tokens_w_non_text = (
        num_prompt_tokens_wo_non_text + await nt_block.aestimate_tokens()
    )
    available_context_size_w_non_text = prompt_helper._get_available_context_size(
        expected_num_prompt_tokens_w_non_text
    )
    expected_chunk_size_w_non_text = (
        available_context_size_w_non_text // num_chunks - padding
    )

    # Sanity checks
    assert await nt_block.aestimate_tokens() > 0

    #
    assert chunk_size_w_non_text == expected_chunk_size_w_non_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "non_text_block",
    [
        ImageBlock(image=b"{image_bytes}"),
        AudioBlock(audio=b"{audio_bytes}"),
        VideoBlock(video=b"{video_bytes}"),
        DocumentBlock(data=b"{pdf_bytes}"),
    ],
)
async def test_chat_prompt_helper_aget_available_chunk_size_does_not_count_unformatted_non_text_blocks(
    non_text_block,
):
    """
    Test that non-text blocks are counted correctly.

    chunk_size = (context_window - num_prompt_tokens - self.num_output) // num_chunks - padding

    In this test, the difference in num_prompt_tokens when including a non-text block should be equal to calling
    formatted non_text_block.estimate_tokens().
    """
    text = "Tell me a joke that is thematic to this image/audio/video."
    prompt_wo_non_text = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system", blocks=[TextBlock(text="You're a a helpful assistant.")]
            ),
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text=text),
                ],
            ),
        ]
    )

    prompt_w_non_text = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system", blocks=[TextBlock(text="You're a a helpful assistant.")]
            ),
            ChatMessage(role="user", blocks=[TextBlock(text=text), non_text_block]),
        ]
    )

    prompt_helper = ChatPromptHelper(
        context_window=4096,
        num_output=512,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # Chunk size limit set trivially to context window size so algebra is simpler
        chunk_size_limit=4096,
    )

    chunk_size_wo_non_text = await prompt_helper._aget_available_chunk_size(
        prompt_wo_non_text, num_chunks=5, padding=0
    )
    chunk_size_w_non_text = await prompt_helper._aget_available_chunk_size(
        prompt_w_non_text, num_chunks=5, padding=0
    )

    # They should be the same since non_text_block is unformatted
    assert chunk_size_w_non_text == chunk_size_wo_non_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_prompt",
    [
        ChatPromptTemplate(message_templates=[ChatMessage(content="test{text}")]),
        RichPromptTemplate(
            template_str="""{% chat role="user" %}\ntest{{text}}\n{% endchat %}"""
        ),
    ],
)
async def test_chat_prompt_helper_atruncate_text(test_prompt):
    """Test truncate for ChatPromptTemplate and RichPromptTemplate objects only containing text."""
    # set context_window=23
    # For each message, there's 4 tokens for content + 5 for the padding + 1 for role + 3 per message
    prompt_helper = ChatPromptHelper(
        context_window=23, num_output=0, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    text_messages = [
        ChatMessage(content="This is a test foo bar"),
        ChatMessage(content="Hello world bar foo"),
    ]

    truncated_messages = await prompt_helper.atruncate(
        prompt=test_prompt, messages=text_messages
    )
    assert truncated_messages == [
        ChatMessage(content="This is a test"),
        ChatMessage(content="Hello world bar foo"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mm_prompt",
    [
        ChatPromptTemplate.from_messages(
            [
                ChatMessage(
                    role="system",
                    blocks=[TextBlock(text="You are a a helpful assistant.")],
                ),
                ChatMessage(
                    role="user",
                    blocks=[
                        TextBlock(text="Describe the following content."),
                        ImageBlock(image=b"{image_bytes}"),
                        AudioBlock(audio=b"{audio_bytes}"),
                        VideoBlock(video=b"{video_bytes}"),
                        DocumentBlock(data=b"{pdf_bytes}"),
                    ],
                ),
            ]
        ),
        RichPromptTemplate(
            template_str=(
                """{% chat role="system" %}\nYou are a a helpful assistant.\n{% endchat %}"""
                """{% chat role="user" %}\nDescribe the following content.\n"""
                """{% for message in context %}\n"""
                """{% for block in message.blocks %}\n"""
                """{% if block.block_type == 'text' %}\n"""
                """{{ block.text }}\n"""
                """{% elif block.block_type == 'image' %}\n"""
                """{{ block.image | image }}\n"""
                """{% elif block.block_type == 'audio' %}\n"""
                """{{ block.audio | audio }}\n"""
                """{% elif block.block_type == 'video' %}\n"""
                """{{ block.video | video }}\n"""
                """{% elif block.block_type == 'document' %}\n"""
                """{{ block.data | document }}\n"""
                """{% endif %}\n"""
                """{% endfor %}\n"""
                """{% endfor %}\n"""
                """{% endchat %}"""
            )
        ),
    ],
)
async def test_chat_prompt_helper_atruncate_multimodal(
    mm_prompt, png_1px, mock_pdf_bytes
):
    """
    Test truncate for ChatPromptTemplate and RichPromptTemplate objects containing multimodal content.
    """
    tb = TextBlock(text="This is a test foo bar")
    ib = ImageBlock(image=png_1px)
    ab = AudioBlock(audio=b"fake_audio")
    vb = VideoBlock(video=b"fake_video")
    db = DocumentBlock(data=mock_pdf_bytes)
    messages = [
        # Messages are intentionally in order of increasing size
        # tb.estimate_tokens() == 6
        # ab.estimate_tokens() == 256
        # ib.estimate_tokens() == 258
        # db.estimate_tokens() == 512
        # vb.estimate_tokens() == 2048
        ChatMessage(blocks=[tb, ab, ib, db, vb]),
        # Note that the second message does not contain an text block to start
        ChatMessage(blocks=[ab, ib, db, vb]),
        # Note that the third block does not contain an text block or the image block to start
        ChatMessage(blocks=[ib, db, vb]),
        ChatMessage(blocks=[db, vb]),
        ChatMessage(blocks=[vb]),
    ]

    # num base prompt tokens = 20
    # Based on:
    # prompt_messages = get_empty_prompt_messages(mm_prompt)
    # num_prompt_tokens = prompt_helper._token_counter.estimate_tokens_in_messages(prompt_messages)

    prompt_helper1 = ChatPromptHelper(
        # Both chunks are allotted enough space for the text block only
        context_window=20 + await tb.aestimate_tokens() * len(messages),
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
    )
    truncated_messages1 = await prompt_helper1.atruncate(
        prompt=mm_prompt, messages=messages, padding=0
    )
    truncated_messages1_strict = await prompt_helper1.atruncate(
        prompt=mm_prompt, messages=messages, padding=0, strict=True
    )

    prompt_helper2 = ChatPromptHelper(
        # Both chunks are allotted enough space for the text and audio blocks
        context_window=20
        + (await tb.aestimate_tokens() + await ab.aestimate_tokens()) * len(messages),
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
    )
    truncated_messages2 = await prompt_helper2.atruncate(
        prompt=mm_prompt, messages=messages, padding=0
    )
    truncated_messages2_strict = await prompt_helper2.atruncate(
        prompt=mm_prompt, messages=messages, padding=0, strict=True
    )

    prompt_helper3 = ChatPromptHelper(
        # Both chunks are allotted enough space for text all tokens
        context_window=20
        + sum([await b.aestimate_tokens() for b in [tb, ab, ib, db, vb]])
        * len(messages),
        num_output=0,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
    )
    truncated_messages3 = await prompt_helper3.atruncate(
        prompt=mm_prompt, messages=messages, padding=0
    )
    truncated_messages3_strict = await prompt_helper3.atruncate(
        prompt=mm_prompt, messages=messages, padding=0, strict=True
    )

    assert truncated_messages1 == [
        # First message should be truncated to just the text block
        ChatMessage(blocks=[tb]),
        # The allotted tokens are insufficient to include the audio block (which cannot be split without ffmpeg),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the audio block is included in its entirety.
        ChatMessage(blocks=[ab]),
        # The allotted tokens are insufficient to include the image block (which cannot be split),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the image block is included in its entirety.
        ChatMessage(blocks=[ib]),
        # The allotted tokens are sufficient to include the document block (which cannot be split currently),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the document block is included in its entirety.
        ChatMessage(blocks=[db]),
        # The allotted tokens are sufficient to include the video block (which cannot be split without ffmpeg),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the video block is included in its entirety.
        ChatMessage(blocks=[vb]),
    ]

    assert truncated_messages1_strict == [
        # First message should be truncated to just the text block
        ChatMessage(blocks=[tb]),
        # The remaining messages are truncated to be empty so strict truncation results in no messages
    ]

    assert truncated_messages2 == [
        # First message should be truncated to just the text and audio blocks
        ChatMessage(blocks=[tb, ab]),
        # Second message should be truncated to just the audio block
        ChatMessage(blocks=[ab]),
        # The allotted tokens are insufficient to include the image block (which cannot be split),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the image block is included in its entirety.
        ChatMessage(blocks=[ib]),
        # The allotted tokens are sufficient to include the document block (which cannot be split currently),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the document block is included in its entirety.
        ChatMessage(blocks=[db]),
        # The allotted tokens are sufficient to include the video block (which cannot be split without ffmpeg),
        # However, since there is no prior block and default truncation mode is strict = False,
        # the video block is included in its entirety.
        ChatMessage(blocks=[vb]),
    ]

    assert truncated_messages2_strict == [
        # First message should be truncated to just the text and audio blocks
        ChatMessage(blocks=[tb, ab]),
        # Second message should be truncated to just the audio block
        ChatMessage(blocks=[ab]),
        # The remaining messages are truncated to be empty so strict truncation results in no messages
    ]

    assert truncated_messages3 == messages
    assert truncated_messages3_strict == messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_prompt",
    [
        ChatPromptTemplate(
            message_templates=[ChatMessage(content="This is the prompt{text}")]
        ),
        RichPromptTemplate(
            template_str="""{% chat role="user" %}\nThis is the prompt{{text}}\n{% endchat %}"""
        ),
    ],
)
async def test_chat_prompt_helper_arepack_text(test_prompt):
    """Test repack for ChatPromptTemplate and RichPromptTemplate with text only messages."""
    prompt_helper = ChatPromptHelper(
        context_window=17,
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # TODO: Look into separators for text chunks?
        # separator="\n\n",
    )
    text_chunks = ["Hello", "world", "foo", "Hello", "world", "bar"]
    text_messages = [ChatMessage(content=content) for content in text_chunks]
    compacted_chunks = await prompt_helper.arepack(test_prompt, text_messages)
    assert compacted_chunks == [
        ChatMessage(content="Hello world foo"),
        ChatMessage(content="Hello world bar"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mm_prompt_template",
    [
        ChatPromptTemplate.from_messages(
            [
                ChatMessage(
                    role="user",
                    blocks=[
                        ImageBlock(image=b"{image_bytes}"),
                        TextBlock(text="This is the prompt{text}"),
                    ],
                )
            ]
        ),
        RichPromptTemplate(
            template_str=(
                """{% chat role="user" %}\n"""
                """{{ image_url | image }}\n"""
                """This is the prompt{{text}}\n"""
                """{% endchat %}"""
            )
        ),
    ],
)
async def test_chat_prompt_helper_arepack_multimodal(mm_prompt_template, png_1px_b64):
    """Test repack for ChatPromptTemplate and RichPromptTemplate with multimodal messages."""
    # Hack for working around banks dependency issues
    if isinstance(mm_prompt_template, ChatPromptTemplate):
        mm_prompt = mm_prompt_template.partial_format(image_bytes=png_1px_b64)
    else:
        mm_prompt = mm_prompt_template.partial_format(
            image_url="data:image/png;base64," + png_1px_b64.decode()
        )
    ib = ImageBlock(image=png_1px_b64)

    prompt_helper1 = ChatPromptHelper(
        context_window=17,  # Not adjusted for image block
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # separator="\n\n",
    )

    prompt_helper2 = ChatPromptHelper(
        context_window=17 + await ib.aestimate_tokens(),  # Adjusted for image block
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # separator="\n\n",
    )

    text_messages = [
        ChatMessage(role="user", content=content)
        for content in ["Hello", "world", "foo", "Hello", "world", "bar"]
    ]
    with pytest.raises(ValueError):
        # Not enough space for text when accounting for image block
        await prompt_helper1.arepack(mm_prompt, text_messages)

    compacted_chunks = await prompt_helper2.arepack(mm_prompt, text_messages)
    assert compacted_chunks == [
        ChatMessage(blocks=[TextBlock(text="Hello world foo")]),
        ChatMessage(blocks=[TextBlock(text="Hello world bar")]),
    ]


@pytest.mark.asyncio
async def test_chat_prompt_helper_chat_prompt_template_repack_arbitrary(
    png_1px, mock_pdf_bytes
):
    """
    Test repack for RichPromptTemplate with arbitrary content.

    This is currently only supported for RichPromptTemplate since ChatPromptTemplate does not support
    a message/ContentBlock which could be any of text, image, audio, video, document, etc.
    """
    mm_prompt = RichPromptTemplate(
        template_str=(
            """{% chat role="system" %}\nYou are a a helpful assistant.\n{% endchat %}"""
            """{% chat role="user" %}\nDescribe the following content.\n"""
            """{% for block in context.blocks %}\n"""
            """{% if block.block_type == 'text' %}\n"""
            """{{ block.text }}\n"""
            """{% elif block.block_type == 'image' %}\n"""
            """{{ block.inline_url() | image }}\n"""
            """{% elif block.block_type == 'audio' %}\n"""
            """{{ block.inline_url() | audio }}\n"""
            """{% elif block.block_type == 'video' %}\n"""
            """{{ block.inline_url() | video }}\n"""
            """{% elif block.block_type == 'document' %}\n"""
            """{{ block.inline_url() | document }}\n"""
            """{% endif %}\n"""
            """{% endfor %}\n"""
            """{% endchat %}"""
        )
    )
    tb = TextBlock(text="Hello world foo")
    ib = ImageBlock(image=png_1px)
    ab = AudioBlock(audio=b"fake_audio", format="mp3")
    vb = VideoBlock(video=b"fake_video", video_mimetype="video/mp4")
    db = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    messages = [
        ChatMessage(blocks=[tb]),
        ChatMessage(blocks=[ib]),
        ChatMessage(blocks=[ab]),
        ChatMessage(blocks=[vb]),
        ChatMessage(blocks=[db]),
    ]

    prompt_helper1 = ChatPromptHelper(
        context_window=8129,
        num_output=512,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # separator="\n\n",
    )

    prompt_helper2 = ChatPromptHelper(
        context_window=4096,
        num_output=512,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        # separator="\n\n",
    )

    compacted_chunks1 = await prompt_helper1.arepack(mm_prompt, messages)
    compacted_chunks2 = await prompt_helper2.arepack(mm_prompt, messages)
    formatted_prompts1 = [
        mm_prompt.format_messages(context=compacted_chunks)
        for compacted_chunks in compacted_chunks1
    ]
    formatted_prompts2 = [
        mm_prompt.format_messages(context=compacted_chunks)
        for compacted_chunks in compacted_chunks2
    ]

    # Combines all messages into one since enough space
    assert compacted_chunks1 == [ChatMessage(blocks=[tb, ib, ab, vb, db])]
    # Splits into two messages due to space constraints
    assert compacted_chunks2 == [
        ChatMessage(blocks=[tb, ib, ab]),
        ChatMessage(blocks=[vb, db]),
    ]

    # 1 prompt with 1 System message + 1 user message with all the content blocks packed in
    assert len(formatted_prompts1) == 1
    assert len(formatted_prompts1[0]) == 2
    assert formatted_prompts1[0][0] == ChatMessage(
        role="system", blocks=[TextBlock(text="You are a a helpful assistant.")]
    )
    assert formatted_prompts1[0][1].role == "user"
    # Consecutive text blocks should be merged into one
    assert formatted_prompts1[0][1].blocks[0] == TextBlock(
        text="Describe the following content.\nHello world foo"
    )
    # Rich prompt template populates the URL fields for non text blocks, instead of raw bytes.
    # So, while the blocks are not equal in an object sense (different attributes), they do contain the same data
    assert formatted_prompts1[0][1].blocks[1] != ib
    assert (
        formatted_prompts1[0][1].blocks[1].resolve_image().read()
        == ib.resolve_image().read()
    )
    assert formatted_prompts1[0][1].blocks[2] != ab
    assert (
        formatted_prompts1[0][1].blocks[2].resolve_audio().read()
        == ab.resolve_audio().read()
    )
    assert formatted_prompts1[0][1].blocks[3] != vb
    assert (
        formatted_prompts1[0][1].blocks[3].resolve_video().read()
        == vb.resolve_video().read()
    )
    assert formatted_prompts1[0][1].blocks[4] != ab
    assert (
        formatted_prompts1[0][1].blocks[4].resolve_document().read()
        == db.resolve_document().read()
    )

    # 2 prompts due to space constraints
    # 1 with the System message + 1 user message with text, image and audio blocks,
    # 1 with the System message + 1 user message with text, video and document blocks
    assert len(formatted_prompts2) == 2
    assert all(len(formatted_prompt) == 2 for formatted_prompt in formatted_prompts2)
    assert all(
        formateted_prompt[0]
        == ChatMessage(
            role="system", blocks=[TextBlock(text="You are a a helpful assistant.")]
        )
        for formateted_prompt in formatted_prompts2
    )
    # First prompt should contain text, image, and audio blocks
    assert formatted_prompts2[0][1].role == "user"
    assert formatted_prompts2[0][1].blocks[0] == TextBlock(
        text="Describe the following content.\nHello world foo"
    )
    assert formatted_prompts2[0][1].blocks[1] != ib
    assert (
        formatted_prompts2[0][1].blocks[1].resolve_image().read()
        == ib.resolve_image().read()
    )
    assert formatted_prompts2[0][1].blocks[2] != ab
    assert (
        formatted_prompts2[0][1].blocks[2].resolve_audio().read()
        == ab.resolve_audio().read()
    )
    # Second prompt should contain prompt text, video, and document blocks
    assert formatted_prompts2[1][1].role == "user"
    assert formatted_prompts2[1][1].blocks[0] == TextBlock(
        text="Describe the following content."
    )
    assert formatted_prompts2[1][1].blocks[1] != vb
    assert (
        formatted_prompts2[1][1].blocks[1].resolve_video().read()
        == vb.resolve_video().read()
    )
    assert formatted_prompts2[1][1].blocks[2] != db
    assert (
        formatted_prompts2[1][1].blocks[2].resolve_document().read()
        == db.resolve_document().read()
    )


def test_prompt_helper_get_biggest_chat_prompt_text() -> None:
    """Test get_biggest_chat_prompt from PromptHelper."""
    prompt1 = ChatPromptTemplate(
        message_templates=[ChatMessage(content="This is the prompt{text}")]
    )
    prompt2 = ChatPromptTemplate(
        message_templates=[ChatMessage(content="This is the longer prompt{text}")]
    )
    prompt3 = ChatPromptTemplate(
        message_templates=[ChatMessage(content="This is the {text}")]
    )
    biggest_prompt = get_biggest_chat_prompt([prompt1, prompt2, prompt3])

    assert biggest_prompt == prompt2


def test_prompt_helper_get_biggest_chat_prompt_text_multimodal(
    png_1px, mock_pdf_bytes
) -> None:
    """Test get_biggest_chat_prompt from PromptHelper."""
    prompt1 = ChatPromptTemplate(
        message_templates=[
            ChatMessage(blocks=[TextBlock(text="This is the prompt{text}")])
        ]
    )
    prompt2 = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                blocks=[
                    TextBlock(text="This is the prompt{text}"),
                    ImageBlock(image=png_1px),
                ]
            )
        ]
    )
    prompt3 = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                blocks=[
                    TextBlock(text="This is the prompt{text}"),
                    ImageBlock(image=png_1px),
                    DocumentBlock(data=mock_pdf_bytes),
                ]
            )
        ]
    )
    assert prompt3 == get_biggest_chat_prompt([prompt1, prompt2, prompt3])
    assert prompt2 == get_biggest_chat_prompt([prompt1, prompt2])


def test_chat_prompt_helper_json_in_prompt() -> None:
    """Test that a JSON object in the prompt doesn't cause an error."""
    # test with chat messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system",
                content="You are a helpful assistant that knows about {topic}. Here's some JSON: {'foo': 'bar'}",
            ),
            ChatMessage(
                role="user",
                content="What is the capital of the moon? Here's some JSON: {'foo': 'bar'}",
            ),
        ]
    )
    prompt.partial_format(topic="the moon")
    prompt_helper = ChatPromptHelper()

    texts = prompt_helper.repack(prompt, [ChatMessage(content="hello_world")])
    assert len(texts) == 1
    assert prompt.template_vars == ["topic"]

    # test with more complex JSON
    prompt = ChatPromptTemplate.from_messages(
        [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful assistant that knows about {topic}\n"
                    "Here's some JSON: {'foo': 'bar'}\n"
                    "here's some other stuff: {key: val for val in d.items()}\n"
                ),
            ),
            ChatMessage(
                role="user",
                content="What is the capital of the moon? Here's some JSON: {'foo': 'bar'}",
            ),
        ]
    )
    prompt.partial_format(topic="the moon")
    prompt_helper = ChatPromptHelper()

    texts = prompt_helper.repack(prompt, [ChatMessage(content="hello_world")])
    assert len(texts) == 1
    assert prompt.template_vars == ["topic"]
