"""Test prompts."""

from typing import List

import pytest

from gpt_index.prompts.base import Prompt


class TestPrompt(Prompt):
    """Test prompt class."""

    input_variables: List[str] = ["text", "foo"]


def test_prompt_validate() -> None:
    """Test prompt validate."""
    # assert passes
    prompt_txt = "hello {text} {foo}"
    TestPrompt(prompt_txt)

    # assert fails (missing required values)
    with pytest.raises(ValueError):
        prompt_txt = "hello {tmp}"
        TestPrompt(prompt_txt)

    # assert fails (extraneous values)
    with pytest.raises(ValueError):
        prompt_txt = "hello {text} {foo} {text2}"
        TestPrompt(prompt_txt)


def test_partial_format() -> None:
    """Test partial format."""
    prompt_txt = "hello {text} {foo}"
    prompt = TestPrompt(prompt_txt)

    prompt_fmt = prompt.partial_format(foo="bar")

    assert isinstance(prompt_fmt, TestPrompt)
    assert prompt_fmt.format(text="world") == "hello world bar"


def test_from_prompt() -> None:
    """Test new prompt from a partially formatted prompt."""

    class TestPromptTextOnly(Prompt):
        """Test prompt class."""

        input_variables: List[str] = ["text"]

    prompt_txt = "hello {text} {foo}"
    prompt = TestPrompt(prompt_txt)
    prompt_fmt = prompt.partial_format(foo="bar")

    prompt_new = TestPromptTextOnly.from_prompt(prompt_fmt)
    assert isinstance(prompt_new, TestPromptTextOnly)

    assert prompt_new.format(text="world2") == "hello world2 bar"
