from llama_index.agent.react.output_parser import (
    extract_final_response,
    extract_tool_use,
)


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


def test_extract_final_response() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2
"""

    expected_thought = (
        "I have enough information to answer the question "
        "without using any more tools."
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
        "I have enough information to answer the question "
        "without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert (
        answer
        == """Here is the answer:

This is the second line."""
    )
