from llama_index.core import PromptTemplate
from llama_index.core.agent.react.prompts import (
    CONTEXT_REACT_CHAT_SYSTEM_HEADER,
    REACT_CHAT_SYSTEM_HEADER,
)
from llama_index.core.agent.workflow import ReActAgent

from textwrap import dedent


def test_partial_formatted_system_prompt():
    """Partially formatted context should be preserved."""
    agent = ReActAgent()

    prompt = PromptTemplate(
        dedent(
            """\
            Required template variables:
            {tool_desc}
            {tool_names}

            Additional variables:
            {dummy_var}
            """
        )
    )

    dummy_var = "dummy_context"
    agent.update_prompts({"react_header": prompt.partial_format(dummy_var=dummy_var)})

    assert dummy_var in agent.formatter.system_header


def test_system_prompt_passed_to_formatter():
    """system_prompt should be set as formatter context and use the context-aware template."""
    system_prompt = "You are a helpful financial advisor."
    agent = ReActAgent(system_prompt=system_prompt)

    assert agent.formatter.context == system_prompt
    assert "{context}" in agent.formatter.system_header
    assert agent.formatter.system_header == CONTEXT_REACT_CHAT_SYSTEM_HEADER

    # Verify the context actually appears in the formatted output
    formatted = agent.formatter.format(tools=[], chat_history=[])
    assert system_prompt in formatted[0].content


def test_no_system_prompt_uses_default_template():
    """Without system_prompt the formatter should use the default template (no context placeholder)."""
    agent = ReActAgent()

    assert agent.formatter.context == ""
    assert agent.formatter.system_header == REACT_CHAT_SYSTEM_HEADER


def test_system_prompt_with_custom_formatter_context():
    """system_prompt should be prepended to an existing formatter context."""
    from llama_index.core.agent.react.formatter import ReActChatFormatter

    custom_context = "Extra context from user."
    formatter = ReActChatFormatter.from_defaults(context=custom_context)
    system_prompt = "You are a helpful assistant."
    agent = ReActAgent(system_prompt=system_prompt, formatter=formatter)

    assert system_prompt in agent.formatter.context
    assert custom_context in agent.formatter.context
    assert "{context}" in agent.formatter.system_header
