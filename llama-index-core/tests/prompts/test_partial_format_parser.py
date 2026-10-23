from threading import Lock
from typing import Any

import pytest

from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.core.types import BaseOutputParser


class LockedOutputParser(BaseOutputParser):
    def __init__(self) -> None:
        self.lock = Lock()

    def parse(self, output: str) -> Any:
        return output

    def format(self, query: str) -> str:
        return query + "\nReturn JSON."


@pytest.mark.parametrize("is_chat", [False, True])
def test_partial_format_preserves_uncopyable_parser(is_chat: bool) -> None:
    parser = LockedOutputParser()
    prompt = (
        ChatPromptTemplate.from_messages(
            [("user", "Hello {name} {topic}")], output_parser=parser
        )
        if is_chat
        else PromptTemplate("Hello {name} {topic}", output_parser=parser)
    )

    partial = prompt.partial_format(name="Ada")

    assert partial is not prompt
    assert partial.output_parser is parser
    assert prompt.output_parser is parser
    assert prompt.kwargs == {}
    assert partial.kwargs == {"name": "Ada"}
    assert partial.format_messages(topic="graphs")[0].content == (
        "Hello Ada graphs\nReturn JSON."
    )
    assert prompt.format_messages(name="Grace", topic="prompts")[0].content == (
        "Hello Grace prompts\nReturn JSON."
    )


@pytest.mark.parametrize("is_chat", [False, True])
def test_partial_format_failure_does_not_drop_parser(is_chat: bool) -> None:
    parser = LockedOutputParser()
    metadata = {"uncopyable": Lock()}
    prompt = (
        ChatPromptTemplate.from_messages(
            [("user", "Hello {name}")], output_parser=parser, metadata=metadata
        )
        if is_chat
        else PromptTemplate("Hello {name}", output_parser=parser, metadata=metadata)
    )

    with pytest.raises(TypeError, match="cannot pickle"):
        prompt.partial_format(name="Ada")

    assert prompt.output_parser is parser
    assert prompt.kwargs == {}
    assert prompt.format_messages(name="Grace")[0].content == (
        "Hello Grace\nReturn JSON."
    )
