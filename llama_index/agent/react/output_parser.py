"""ReAct output parser."""


from llama_index.output_parsers.utils import extract_json_str
from llama_index.types import BaseOutputParser
from typing import Tuple
from llama_index.agent.react.types import (
    BaseReasoningStep,
    ResponseReasoningStep,
    ActionReasoningStep,
)
import ast

import re


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    pattern = r"\s*Thought:(.*?)Action:(.*?)Action Input:(.*?)(?:\n|$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not extract tool use from input text: {}".format(input_text)
        )

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def extract_final_response(input_text: str) -> Tuple[str, str]:
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:\n|$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not extract final answer from input text: {}".format(input_text)
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str) -> BaseReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="I can answer without any tools.", response=output
            )

        if "Answer:" in output:
            thought, answer = extract_final_response(output)
            return ResponseReasoningStep(thought=thought, response=answer)

        if "Action:" in output:
            thought, action, action_input = extract_tool_use(output)
            json_str = extract_json_str(action_input)

            # NOTE: we found that json.loads does not reliably parse
            # json with single quotes, so we use ast instead
            # action_input_dict = json.loads(json_str)
            action_input_dict = ast.literal_eval(json_str)

            return ActionReasoningStep(
                thought=thought, action=action, action_input=action_input_dict
            )

        raise ValueError("Could not parse output: {}".format(output))

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
