"""Test PromptHelper."""

from typing import Optional, Type, Union

import pytest
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser.text.utils import truncate_text
from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_utils import (
    get_biggest_prompt,
    get_empty_prompt_txt,
)
from llama_index.core.schema import TextNode
from tests.mock_utils.mock_utils import mock_tokenizer


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
def test_get_chunk_size(
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


def test_get_text_splitter() -> None:
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


def test_get_text_splitter_partial() -> None:
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


def test_truncate() -> None:
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


def test_get_numbered_text_from_nodes() -> None:
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


def test_repack() -> None:
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


def test_get_biggest_prompt() -> None:
    """Test get_biggest_prompt from PromptHelper."""
    prompt1 = PromptTemplate("This is the prompt{text}")
    prompt2 = PromptTemplate("This is the longer prompt{text}")
    prompt3 = PromptTemplate("This is the {text}")
    biggest_prompt = get_biggest_prompt([prompt1, prompt2, prompt3])

    assert biggest_prompt == prompt2


def test_json_in_prompt() -> None:
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
