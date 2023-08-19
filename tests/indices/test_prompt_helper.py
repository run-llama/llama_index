"""Test PromptHelper."""
from typing import cast

from llama_index.bridge.langchain import PromptTemplate as LangchainPrompt
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_utils import get_biggest_prompt, get_empty_prompt_txt
from llama_index.schema import TextNode
from llama_index.text_splitter.utils import truncate_text
from tests.mock_utils.mock_utils import mock_tokenizer


def test_get_chunk_size() -> None:
    """Test get chunk size given prompt."""
    # test with 1 chunk
    prompt = PromptTemplate("This is the prompt")
    prompt_helper = PromptHelper(
        context_window=11, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 1, padding=0)
    assert chunk_size == 6

    # test having 2 chunks
    prompt_helper = PromptHelper(
        context_window=11, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=0)
    assert chunk_size == 3

    # test with 2 chunks, and with chunk_size_limit
    prompt_helper = PromptHelper(
        context_window=11,
        num_output=1,
        chunk_overlap_ratio=0,
        tokenizer=mock_tokenizer,
        chunk_size_limit=2,
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=0)
    assert chunk_size == 2

    # test padding
    prompt_helper = PromptHelper(
        context_window=11, num_output=1, chunk_overlap_ratio=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=1)
    assert chunk_size == 2


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
    assert text_splitter._chunk_size == 2
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
