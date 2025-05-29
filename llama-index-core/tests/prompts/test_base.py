"""Test prompts."""


from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.types import BaseOutputParser


class MockOutputParser(BaseOutputParser):
    """Mock output parser."""

    def __init__(self, format_string: str) -> None:
        self._format_string = format_string

    def parse(self, output: str) -> Any:
        return {"output": output}

    def format(self, query: str) -> str:
        return query + "\n" + self._format_string


@pytest.fixture()
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


def test_template_var_mappings() -> None:
    """Test template variable mappings."""
    qa_prompt_tmpl = """\
Here's some context:
{foo}
Given the context, please answer the final question:
{bar}
"""
    template_var_mappings = {
        "context_str": "foo",
        "query_str": "bar",
    }
    # try regular prompt template
    qa_prompt = PromptTemplate(
        qa_prompt_tmpl, template_var_mappings=template_var_mappings
    )
    fmt_prompt = qa_prompt.format(query_str="abc", context_str="def")
    assert (
        fmt_prompt
        == """\
Here's some context:
def
Given the context, please answer the final question:
abc
"""
    )
    # try partial format
    qa_prompt_partial = qa_prompt.partial_format(query_str="abc2")
    fmt_prompt_partial = qa_prompt_partial.format(context_str="def2")
    assert (
        fmt_prompt_partial
        == """\
Here's some context:
def2
Given the context, please answer the final question:
abc2
"""
    )

    # try chat prompt template
    # partial template var mapping
    template_var_mappings = {
        "context_str": "foo",
        "query_str": "bar",
    }
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content="This is a system message with a {sys_param}",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content="hello {foo} {bar}", role=MessageRole.USER),
        ],
        prompt_type=PromptType.CONVERSATION,
        template_var_mappings=template_var_mappings,
    )
    fmt_prompt = chat_template.format(
        query_str="abc", context_str="def", sys_param="sys_arg"
    )
    assert fmt_prompt == (
        "system: This is a system message with a sys_arg\n"
        "user: hello def abc\n"
        "assistant: "
    )


def test_function_mappings() -> None:
    """Test function mappings."""
    test_prompt_tmpl = """foo bar {abc} {xyz}"""

    ## PROMPT 1
    # test a format function that uses values of both abc and def
    def _format_abc(**kwargs: Any) -> str:
        """Given kwargs, output formatted variable."""
        return f"{kwargs['abc']}-{kwargs['xyz']}"

    test_prompt = PromptTemplate(
        test_prompt_tmpl, function_mappings={"abc": _format_abc}
    )
    assert test_prompt.format(abc="123", xyz="456") == "foo bar 123-456 456"

    # test partial
    test_prompt_partial = test_prompt.partial_format(xyz="456")
    assert test_prompt_partial.format(abc="789") == "foo bar 789-456 456"

    ## PROMPT 2
    # test a format function that only depends on values of xyz
    def _format_abc_2(**kwargs: Any) -> str:
        """Given kwargs, output formatted variable."""
        return f"{kwargs['xyz']}"

    test_prompt_2 = PromptTemplate(
        test_prompt_tmpl, function_mappings={"abc": _format_abc_2}
    )
    assert test_prompt_2.format(xyz="456") == "foo bar 456 456"

    # test that formatting abc itself will throw an error
    with pytest.raises(KeyError):
        test_prompt_2.format(abc="123")

    ## PROMPT 3 - test prompt with template var mappings
    def _format_prompt_key1(**kwargs: Any) -> str:
        """Given kwargs, output formatted variable."""
        return f"{kwargs['prompt_key1']}-{kwargs['prompt_key2']}"

    template_var_mappings = {
        "prompt_key1": "abc",
        "prompt_key2": "xyz",
    }
    test_prompt_3 = PromptTemplate(
        test_prompt_tmpl,
        template_var_mappings=template_var_mappings,
        # NOTE: with template mappings, needs to use the source variable names,
        # not the ones being mapped to in the template
        function_mappings={"prompt_key1": _format_prompt_key1},
    )
    assert (
        test_prompt_3.format(prompt_key1="678", prompt_key2="789")
        == "foo bar 678-789 789"
    )

    ### PROMPT 4 - test chat prompt template
    chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                content="This is a system message with a {sys_param}",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content="hello {abc} {xyz}", role=MessageRole.USER),
        ],
        prompt_type=PromptType.CONVERSATION,
        function_mappings={"abc": _format_abc},
    )
    fmt_prompt = chat_template.format(abc="tmp1", xyz="tmp2", sys_param="sys_arg")
    assert fmt_prompt == (
        "system: This is a system message with a sys_arg\n"
        "user: hello tmp1-tmp2 tmp2\n"
        "assistant: "
    )


def test_template_with_json() -> None:
    """Test partial format."""
    prompt_txt = 'hello {text} {foo} {"bar": "baz"}'
    prompt = PromptTemplate(prompt_txt)

    assert prompt.format(foo="foo2", text="world") == 'hello world foo2 {"bar": "baz"}'

    assert prompt.format_messages(foo="foo2", text="world") == [
        ChatMessage(content='hello world foo2 {"bar": "baz"}', role=MessageRole.USER)
    ]

    test_case_2 = PromptTemplate("test {message} {test}")
    assert test_case_2.format(message="message") == "test message {test}"

    test_case_3 = PromptTemplate("test {{message}} {{test}}")
    assert test_case_3.format(message="message", test="test") == "test {message} {test}"


def test_template_has_json() -> None:
    """Test partial format."""
    prompt_txt = (
        'hello {text} {foo} \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    )
    except_prompt = (
        'hello world bar \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    )

    prompt_template = PromptTemplate(prompt_txt)
    template_vars = prompt_template.template_vars
    prompt_fmt = prompt_template.partial_format(foo="bar")
    prompt = prompt_fmt.format(text="world")

    assert isinstance(prompt_fmt, PromptTemplate)
    assert template_vars == ["text", "foo"]
    assert prompt == except_prompt
    assert prompt_fmt.format_messages(text="world") == [
        ChatMessage(content=except_prompt, role=MessageRole.USER)
    ]
