"""
The spoke output parsers can take the output of the spoke LLM and transform it into a more suitable format. Particularly, it can make the spoke aware that collaboration is needed based on the output of LLM so that the spoke can initiate inter-spoke communication. We implement a SpokeParser class here.
"""

import re
from typing import Tuple

from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.output_parsers.utils import extract_json_str
from llama_index.core.types import BaseOutputParser


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    """
    Extract thought, action, and action input from the input text.

    Args:
        input_text (str): The text from which to extract the components.

    Returns:
        Tuple[str, str, str]: Extracted thought, action, and action input.

    Raises:
        ValueError: If the input text does not contain the required information.

    """
    pattern = (
        r"\s*Thought: (.*?)\nAction: ([a-zA-Z0-9_]+).*?\nAction Input: .*?(\{.*\})"
    )

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def action_input_parser(json_str: str) -> dict:
    """
    Parse a JSON-like string into a dictionary.

    Args:
        json_str (str): The JSON-like string to parse.

    Returns:
        dict: The parsed dictionary.

    """
    processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
    pattern = r'"(\w+)":\s*"([^"]*)"'
    matches = re.findall(pattern, processed_string)
    return dict(matches)


def extract_final_response(input_text: str) -> Tuple[str, str]:
    """
    Extract the thought and answer from the input text.

    Args:
        input_text (str): The text from which to extract the components.

    Returns:
        Tuple[str, str]: Extracted thought and answer.

    Raises:
        ValueError: If the input text does not contain the required information.

    """
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract final answer from input text: {input_text}"
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


def parse_action_reasoning_step(output: str) -> ActionReasoningStep:
    """
    Parse an action reasoning step from the LLM output.

    Args:
        output (str): The output text to parse.

    Returns:
        ActionReasoningStep: The parsed action reasoning step.

    """
    # Weaker LLMs may generate ReActAgent steps whose Action Input are horrible JSON strings.
    # `dirtyjson` is more lenient than `json` in parsing JSON strings.
    import dirtyjson as json

    thought, action, action_input = extract_tool_use(output)
    json_str = extract_json_str(action_input)
    # First we try json, if this fails we use ast
    try:
        action_input_dict = json.loads(json_str)
    except Exception:
        action_input_dict = action_input_parser(json_str)
    return ActionReasoningStep(
        thought=thought, action=action, action_input=action_input_dict
    )


class SpokeOutputParser(BaseOutputParser):
    """
    A parser to transform the output of the spoke LLM into a more suitable format and
    to make the spoke aware of the need for collaboration based on the LLM's output.
    """

    def __init__(self, functionality_list, spoke_operator, *args, **kwargs) -> None:
        """
        Initialize the SpokeOutputParser.

        Args:
            functionality_list (list): A list of functionalities supported by the spoke.
            spoke_operator (SpokeOperator): An instance of SpokeOperator to handle functionality requests.

        """
        super().__init__(*args, **kwargs)
        self.functionality_list = functionality_list
        self.spoke_operator = spoke_operator
        self.called_functionalities = {}

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """
        Parse output from the ReAct agent.

        Args:
            output (str): The output from the ReAct agent.
            is_streaming (bool): Whether the output is being streamed.

        Returns:
            BaseReasoningStep: The parsed reasoning step.

        Raises:
            ValueError: If the output cannot be parsed.

        """
        if "Thought:" not in output:
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )

        if "Answer:" in output:
            thought, answer = extract_final_response(output)
            return ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        if "Action:" in output:
            thought, action, action_input = extract_tool_use(output)
            if action in self.functionality_list:
                if action not in self.called_functionalities:
                    (
                        message_type,
                        function_schema,
                    ) = self.spoke_operator.probe_functionality(action)

                    if (
                        message_type != "function_probe_response"
                        or function_schema is None
                    ):
                        message = f"Could not probe {action} functionality. YOU MUST NOT PROBE {action} AGAIN!"

                        return ActionReasoningStep(
                            thought="Use message_spoke to deliver instructions.",
                            action="message_spoke",
                            action_input={"message": message},
                        )
                    self.called_functionalities[action] = {}
                    self.called_functionalities[action]["function_schema"] = (
                        function_schema
                    )
                else:
                    function_schema = self.called_functionalities[action][
                        "function_schema"
                    ]
                    message = f'Use the tool "{action}" with the formatted input strictly following the tool parameter dictionary: "{function_schema!s}"'
                    return ActionReasoningStep(
                        thought="",
                        action="message_spoke",
                        action_input={"message": message},
                    )

                message_type, response = self.spoke_operator.make_request(
                    action, action_input
                )
                if message_type != "app_response":
                    message = f"Could not make request to {action}. YOU MUST NOT REQUEST {action} AGAIN!"
                    return ActionReasoningStep(
                        thought="",
                        action="message_spoke",
                        action_input={"message": message},
                    )
                    # Use message_spoke to deliver instructions.
                message = f"Response from {action}: {response!s}"
                return ActionReasoningStep(
                    thought="",
                    action="message_spoke",
                    action_input={"message": message},
                )

            else:
                return parse_action_reasoning_step(output)

        raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """
        Format a query with structured output formatting instructions.

        Args:
            output (str): The output to format.

        Raises:
            NotImplementedError: As this method is not implemented.

        """
        raise NotImplementedError
