import time

import pytest

from llama_index.core.agent.react.output_parser import (
    extract_final_response,
    extract_tool_use,
    parse_action_reasoning_step,
)


def test_parse_action_reasoning_step() -> None:
    mock_input_text = """\
Thought: Gotta use a tool.
Action: tool
Action Input: {'pages': ['coffee'] /* comment */, 'load_kwargs': {}, 'query_str': ''}, along those lines.
"""
    assert parse_action_reasoning_step(mock_input_text).action_input == {
        "pages": ["coffee"],
        "load_kwargs": {},
        "query_str": "",
    }


def test_extract_tool_use() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_no_thought() -> None:
    mock_input_text = """\
I need to use a tool to help me answer the question.
Action: add
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_multiline() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: add



Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_with_nested_dicts() -> None:
    mock_input_text = """\
Thought: Gotta use a tool.
Action: tool
Action Input: {"a": 1, "b": {}}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "Gotta use a tool."
    assert action == "tool"
    assert action_input == '{"a": 1, "b": {}}'


def test_extract_tool_use_() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: QueryEngineTool({"a": 1, "b": 1})
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_extra_action_output() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add (add two numbers)
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_number() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add2
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add2"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_multiline_action_input() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {
    "a": 1,
    "b": 1
}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert (
        action_input
        == """\
{
    "a": 1,
    "b": 1
}"""
    )


def test_extract_tool_use_spurious_newlines() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: add

Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_with_Chinese_characters() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: 加法

Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "加法"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_final_response() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2
"""

    expected_thought = (
        "I have enough information to answer the question without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert answer == "2"


def test_extract_final_response_multiline_answer() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: Here is the answer:

This is the second line.
"""

    expected_thought = (
        "I have enough information to answer the question without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert (
        answer
        == """Here is the answer:

This is the second line."""
    )


def test_react_output_parser_handles_action_in_answer() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: Answer contains Legislative Action: here is the legislative action content.
"""

    expected_thought = (
        "I have enough information to answer the question without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert (
        answer
        == "Answer contains Legislative Action: here is the legislative action content."
    )


def test_extract_final_response_no_thought() -> None:
    """
    A preamble without the "Thought:" prefix is taken as the thought.

    `extract_tool_use` already accepts this shape (see
    `test_extract_tool_use_no_thought`); the answer path should match it.
    """
    mock_input_text = """\
I have enough information to answer the question without using any more tools.
Answer: 2
"""
    thought, answer = extract_final_response(mock_input_text)
    assert (
        thought
        == "I have enough information to answer the question without using any more tools."
    )
    assert answer == "2"


def test_extract_final_response_answer_only() -> None:
    """An answer with no preceding thought at all still parses."""
    mock_input_text = "Answer: 2\n"

    thought, answer = extract_final_response(mock_input_text)
    assert thought == ""
    assert answer == "2"


def test_extract_final_response_no_answer_raises() -> None:
    """Text without an "Answer:" is still a parse failure."""
    with pytest.raises(ValueError, match="Could not extract final answer"):
        extract_final_response("Thought: I am still thinking about it.\n")


def test_extract_final_response_is_linear_without_answer() -> None:
    """
    Guard against the backtracking class fixed in #22334.

    Matching the thought and the answer in one lazy DOTALL pattern is quadratic
    when "Answer:" never appears: the same 80 KB input took ~42 s that way.
    """
    mock_input_text = "Thought: " + ("x" * 200_000)

    start = time.perf_counter()
    with pytest.raises(ValueError, match="Could not extract final answer"):
        extract_final_response(mock_input_text)
    elapsed = time.perf_counter() - start

    # Measured at well under 1 ms; a quadratic pattern needs minutes at this
    # size, so the bound is loose enough not to flake on a loaded runner.
    assert elapsed < 1.0
