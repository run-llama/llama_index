"""Test PromptHelper."""
from typing import cast

from langchain import PromptTemplate as LangchainPrompt

from llama_index.data_structs.node import Node
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.tree.utils import get_numbered_text_from_nodes
from llama_index.prompts.utils import get_biggest_prompt, get_empty_prompt_txt
from llama_index.prompts.base import Prompt
from tests.mock_utils.mock_utils import mock_tokenizer


def test_get_chunk_size() -> None:
    """Test get chunk size given prompt."""
    # test with 1 chunk
    prompt = Prompt("This is the prompt")
    prompt_helper = PromptHelper(
        max_input_size=11, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 1, padding=0)
    assert chunk_size == 6

    # test having 2 chunks
    prompt_helper = PromptHelper(
        max_input_size=11, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=0)
    assert chunk_size == 3

    # test with 2 chunks, and with chunk_size_limit
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=1,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        chunk_size_limit=2,
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=0)
    assert chunk_size == 2

    # test padding
    prompt_helper = PromptHelper(
        max_input_size=11, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    chunk_size = prompt_helper._get_available_chunk_size(prompt, 2, padding=1)
    assert chunk_size == 2


def test_get_text_splitter() -> None:
    """Test get text splitter."""
    test_prompt_text = "This is the prompt{text}"
    test_prompt = Prompt(test_prompt_text)
    prompt_helper = PromptHelper(
        max_input_size=11, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    assert text_splitter._chunk_size == 2
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = text_splitter.truncate_text(test_text)
    assert truncated_text == "Hello world"

    # test with chunk_size_limit
    prompt_helper = PromptHelper(
        max_input_size=11,
        num_output=1,
        max_chunk_overlap=0,
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
    test_prompt = Prompt(test_prompt_text)
    prompt_helper = PromptHelper(
        max_input_size=11, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = text_splitter.truncate_text(test_text)
    assert truncated_text == "Hello world"

    # test with partially formatting
    test_prompt = Prompt(test_prompt_text)
    test_prompt = test_prompt.partial_format(foo="bar")
    prompt_helper = PromptHelper(
        max_input_size=12, num_output=1, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    assert get_empty_prompt_txt(test_prompt) == "This is the bar prompt"
    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        test_prompt, 2, padding=1
    )
    test_text = "Hello world foo Hello world bar"
    text_chunks = text_splitter.split_text(test_text)
    assert text_chunks == ["Hello world", "foo Hello", "world bar"]
    truncated_text = text_splitter.truncate_text(test_text)
    assert truncated_text == "Hello world"


def test_truncate() -> None:
    """Test truncate."""
    # test prompt uses up one token
    test_prompt_txt = "test{text}"
    test_prompt = Prompt(test_prompt_txt)
    # set max_input_size=19
    # For each text chunk, there's 4 tokens for text + 5 for the padding
    prompt_helper = PromptHelper(
        max_input_size=19, num_output=0, max_chunk_overlap=0, tokenizer=mock_tokenizer
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
    test_prompt = Prompt(test_prompt_txt)
    # set max_input_size=17
    # For each text chunk, there's 3 for text, 5 for padding (including number)
    prompt_helper = PromptHelper(
        max_input_size=17, num_output=0, max_chunk_overlap=0, tokenizer=mock_tokenizer
    )
    node1 = Node(text="This is a test foo bar")
    node2 = Node(text="Hello world bar foo")

    text_splitter = prompt_helper.get_text_splitter_given_prompt(
        prompt=test_prompt,
        num_chunks=2,
    )
    response = get_numbered_text_from_nodes([node1, node2], text_splitter=text_splitter)
    assert str(response) == ("(1) This is a\n\n(2) Hello world bar")


def test_repack() -> None:
    """Test repack."""
    test_prompt_text = "This is the prompt{text}"
    test_prompt = Prompt(test_prompt_text)
    prompt_helper = PromptHelper(
        max_input_size=13,
        num_output=1,
        max_chunk_overlap=0,
        tokenizer=mock_tokenizer,
        separator="\n\n",
    )
    text_chunks = ["Hello", "world", "foo", "Hello", "world", "bar"]
    compacted_chunks = prompt_helper.repack(test_prompt, text_chunks)
    assert compacted_chunks == ["Hello\n\nworld\n\nfoo", "Hello\n\nworld\n\nbar"]


def test_get_biggest_prompt() -> None:
    """Test get_biggest_prompt from PromptHelper."""
    prompt1 = Prompt("This is the prompt{text}")
    prompt2 = Prompt("This is the longer prompt{text}")
    prompt3 = Prompt("This is the {text}")
    biggest_prompt = get_biggest_prompt([prompt1, prompt2, prompt3])

    lc_biggest_template = cast(LangchainPrompt, biggest_prompt.prompt).template
    prompt2_template = cast(LangchainPrompt, prompt2.prompt).template
    assert lc_biggest_template == prompt2_template
