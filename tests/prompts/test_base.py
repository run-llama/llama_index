"""Test prompts."""


from typing import Any
import pytest
from llama_index.bridge.langchain import BaseLanguageModel
from llama_index.bridge.langchain import ConditionalPromptSelector as LangchainSelector
from llama_index.bridge.langchain import FakeListLLM
from llama_index.bridge.langchain import PromptTemplate as LangchainTemplate
from llama_index.llms import MockLLM
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.langchain import LangChainLLM
from llama_index.prompts import (
    ChatPromptTemplate,
    LangchainPromptTemplate,
    PromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.prompts.prompt_type import PromptType
from llama_index.types import BaseOutputParser


class MockOutputParser(BaseOutputParser):
    """Mock output parser."""

    def __init__(self, format_string: str) -> None:
        self._format_string = format_string

    def parse(self, output: str) -> Any:
        return {"output": output}

    def format(self, query: str) -> str:
        return query + "\n" + self._format_string


@pytest.fixture
def output_parser() -> BaseOutputParser:
    return MockOutputParser(format_string="output_instruction")


def test_template() -> None:
    """Test partial format."""
    prompt_txt = "hello {text} {foo}"
    prompt = PromptTemplate(prompt_txt)

    prompt_fmt = prompt.partial_format(foo="bar")
    assert isinstance(prompt_fmt, PromptTemplate)

    assert prompt_fmt.format(text="world") == "hello world bar"

    assert prompt_fmt.format_messages(text="world") == [
        ChatMessage(content="hello world bar", role=MessageRole.USER)
    ]


def test_template_output_parser(output_parser: BaseOutputParser) -> None:
    prompt_txt = "hello {text} {foo}"
    prompt = PromptTemplate(prompt_txt, output_parser=output_parser)

    prompt_fmt = prompt.format(text="world", foo="bar")
    assert prompt_fmt == "hello world bar\noutput_instruction"


def test_chat_template() -> None:
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content="This is a system message with a {sys_param}",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content="hello {text} {foo}", role=MessageRole.USER),
        ],
        prompt_type=PromptType.CONVERSATION,
    )

    partial_template = chat_template.partial_format(sys_param="sys_arg")
    messages = partial_template.format_messages(text="world", foo="bar")

    assert messages[0] == ChatMessage(
        content="This is a system message with a sys_arg", role=MessageRole.SYSTEM
    )

    assert partial_template.format(text="world", foo="bar") == (
        "system: This is a system message with a sys_arg\n"
        "user: hello world bar\n"
        "assistant: "
    )


def test_chat_template_output_parser(output_parser: BaseOutputParser) -> None:
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content="This is a system message with a {sys_param}",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content="hello {text} {foo}", role=MessageRole.USER),
        ],
        prompt_type=PromptType.CONVERSATION,
        output_parser=output_parser,
    )

    messages = chat_template.format_messages(
        text="world", foo="bar", sys_param="sys_arg"
    )
    assert (
        messages[0].content
        == "This is a system message with a sys_arg\noutput_instruction"
    )


def test_selector_template() -> None:
    default_template = PromptTemplate("hello {text} {foo}")
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content="This is a system message with a {sys_param}",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content="hello {text} {foo}", role=MessageRole.USER),
        ],
        prompt_type=PromptType.CONVERSATION,
    )

    selector_template = SelectorPromptTemplate(
        default_template=default_template,
        conditionals=[
            (lambda llm: isinstance(llm, MockLLM), chat_template),
        ],
    )

    partial_template = selector_template.partial_format(text="world", foo="bar")

    prompt = partial_template.format()
    assert prompt == "hello world bar"

    messages = partial_template.format_messages(llm=MockLLM(), sys_param="sys_arg")
    assert messages[0] == ChatMessage(
        content="This is a system message with a sys_arg", role=MessageRole.SYSTEM
    )


def test_langchain_template() -> None:
    lc_template = LangchainTemplate.from_template("hello {text} {foo}")
    template = LangchainPromptTemplate(lc_template)

    template_fmt = template.partial_format(foo="bar")
    assert isinstance(template, LangchainPromptTemplate)

    assert template_fmt.format(text="world") == "hello world bar"

    assert template_fmt.format_messages(text="world") == [
        ChatMessage(content="hello world bar", role=MessageRole.USER)
    ]


def test_langchain_selector_template() -> None:
    lc_llm = FakeListLLM(responses=["test"])
    mock_llm = LangChainLLM(llm=lc_llm)

    def is_mock(llm: BaseLanguageModel) -> bool:
        return llm == lc_llm

    default_lc_template = LangchainTemplate.from_template("hello {text} {foo}")
    conditionals = [
        (is_mock, LangchainTemplate.from_template("hello {text} {foo} mock")),
    ]

    lc_selector = LangchainSelector(
        default_prompt=default_lc_template, conditionals=conditionals
    )
    template = LangchainPromptTemplate(selector=lc_selector)

    template_fmt = template.partial_format(foo="bar")
    assert isinstance(template, LangchainPromptTemplate)

    assert template_fmt.format(llm=mock_llm, text="world") == "hello world bar mock"
