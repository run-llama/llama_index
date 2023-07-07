"""ReAct output parser."""


from llama_index.output_parsers.utils import extract_json_str
from llama_index.types import BaseOutputParser
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
        output_lines = output.strip().split("\n")
        thought = output_lines[0].split("Thought:")[1].strip()

        if "Answer:" in output:
            response = output_lines[1].split("Answer:")[1].strip()
            return ResponseReasoningStep(thought=thought, response=response)

        # parse out action
        action_str = output_lines[1].split("Action:")[1].strip()
        raw_action_input_str = output_lines[2].split("Action Input:")[1].strip()
        json_str = extract_json_str(raw_action_input_str)

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
