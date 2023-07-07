"""ReAct output parser."""


from llama_index.types import BaseOutputParser
import re
from llama_index.agent.react.types import (
    BaseReasoningStep,
    ResponseReasoningStep,
    ActionReasoningStep,
)
import ast


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str) -> BaseReasoningStep:
        """Parse, validate, and correct errors programmatically."""
        output_lines = output.split("\n")
        thought = output_lines[0].split("Thought:")[1].strip()

        if "Answer:" in output:
            response = output_lines[1].split("Answer:")[1].strip()
            return ResponseReasoningStep(thought=thought, response=response)

        # parse out action
        action_str = output_lines[1].split("Action:")[1].strip()
        raw_action_input_str = output_lines[2].split("Action Input:")[1].strip()
        # NOTE: this is copied from llama_index.output_parsers.pydantic
        # which is in turn taken from langchain.output_parsers.pydantic
        match = re.search(
            r"\{.*\}",
            raw_action_input_str.strip(),
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"Could not find JSON in {raw_action_input_str}")
        json_str = match.group()

        # NOTE: we found that json.loads does not reliably parse
        # json with single quotes, so we use ast instead
        # action_input_dict = json.loads(json_str)
        action_input_dict = ast.literal_eval(json_str)

        return ActionReasoningStep(
            thought=thought, action=action_str, action_input=action_input_dict
        )

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
