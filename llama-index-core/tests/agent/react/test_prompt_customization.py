import warnings
from textwrap import dedent
from unittest.mock import MagicMock

from llama_index.core import PromptTemplate
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from llama_index.core.agent.workflow import ReActAgent


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


def test_formatter_with_literal_braces():
    """Formatter should not raise KeyError on strings with literal braces."""
    formatter = ReActChatFormatter(
        system_header='Use JSON like {"input": "hello"}\n{tool_desc}\n{tool_names}'
    )

    mock_tool = MagicMock()
    mock_tool.metadata.name = "test_tool"
    mock_tool.metadata.get_name.return_value = "test_tool"
    mock_tool.metadata.description = "A test tool"
    mock_tool.metadata.fn_schema_str = '{"type": "object"}'

    # Should not raise KeyError
    messages = formatter.format(tools=[mock_tool], chat_history=[])
    assert len(messages) > 0
    assert '{"input": "hello"}' in messages[0].content


def test_default_template_preserves_json_examples():
    """Default template's {{...}} should produce {...} in the formatted output."""
    formatter = ReActChatFormatter()

    mock_tool = MagicMock()
    mock_tool.metadata.name = "test_tool"
    mock_tool.metadata.get_name.return_value = "test_tool"
    mock_tool.metadata.description = "A test tool"
    mock_tool.metadata.fn_schema_str = '{"type": "object"}'

    messages = formatter.format(tools=[mock_tool], chat_history=[])
    content = messages[0].content

    # The default template has {{"input": "hello world", "num_beams": 5}}
    # which should become {"input": "hello world", "num_beams": 5} in the output
    assert '{"input": "hello world", "num_beams": 5}' in content
    # Should NOT contain double braces in the output
    assert '{{"input"' not in content


def test_update_prompts_with_preformatted_template():
    """Pre-formatted template with literal braces should not crash the formatter."""
    agent = ReActAgent()

    # Simulate what the issue reporter does: pre-format the default template
    # This collapses {{ to { and fills placeholders
    preformatted = REACT_CHAT_SYSTEM_HEADER.format(
        tool_desc="{tool_desc}",
        tool_names="{tool_names}",
    )
    # preformatted now contains literal { from collapsed {{

    agent.update_prompts({"react_header": preformatted})

    mock_tool = MagicMock()
    mock_tool.metadata.name = "test_tool"
    mock_tool.metadata.get_name.return_value = "test_tool"
    mock_tool.metadata.description = "A test tool"
    mock_tool.metadata.fn_schema_str = '{"type": "object"}'

    # Should not raise KeyError
    messages = agent.formatter.format(tools=[mock_tool], chat_history=[])
    assert len(messages) > 0


def test_formatter_substitutes_tool_args_with_literal_braces():
    """tool_desc and tool_names should be substituted while literal braces survive.

    This is the exact scenario from the bug report: a user-supplied template
    contains literal JSON examples (single braces) alongside {tool_desc} and
    {tool_names} placeholders.  The old str.format() code raised KeyError here;
    format_string() tolerates the literal braces.
    """
    formatter = ReActChatFormatter(
        system_header=(
            'You have access to:\n{tool_desc}\n'
            'Use tools: {tool_names}\n'
            'Action Input must be JSON, e.g. {"query": "search term"}'
        )
    )

    mock_tool = MagicMock()
    mock_tool.metadata.name = "search"
    mock_tool.metadata.get_name.return_value = "search"
    mock_tool.metadata.description = "Search the web"
    mock_tool.metadata.fn_schema_str = '{"type": "object"}'

    messages = formatter.format(tools=[mock_tool], chat_history=[])
    content = messages[0].content

    # tool_desc and tool_names must be substituted with actual values
    assert "Search the web" in content
    assert "> Tool Name: search" in content

    # Literal JSON braces must survive intact in the output
    assert '{"query": "search term"}' in content


def test_update_prompts_warns_on_missing_placeholders():
    """Updating with a prompt missing required placeholders should warn."""
    agent = ReActAgent()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent.update_prompts({"react_header": "No placeholders here"})
        # Should have warnings for both tool_desc and tool_names
        warning_messages = [str(warning.message) for warning in w]
        assert any("tool_desc" in msg for msg in warning_messages)
        assert any("tool_names" in msg for msg in warning_messages)


def test_update_prompts_no_warning_when_placeholders_present():
    """Updating with a prompt that has required placeholders should not warn."""
    agent = ReActAgent()

    prompt_with_placeholders = "Tools: {tool_desc}\nNames: {tool_names}"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent.update_prompts({"react_header": prompt_with_placeholders})
        warning_messages = [str(warning.message) for warning in w]
        assert not any("tool_desc" in msg for msg in warning_messages)
        assert not any("tool_names" in msg for msg in warning_messages)
