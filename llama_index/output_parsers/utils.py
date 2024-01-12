import contextlib
import json
import re
from typing import Any, List

with contextlib.suppress(ImportError):
    import yaml

from llama_index.output_parsers.base import OutputParserException


def _marshal_llm_to_json(output: str) -> str:
    """
    Extract a substring containing valid JSON or array from a string.

    Args:
        output: A string that may contain a valid JSON object or array surrounded by
        extraneous characters or information.

    Returns:
        A string containing a valid JSON object or array.
    """
    output = output.strip().replace("{{", "{").replace("}}", "}")

    left_square = output.find("[")
    left_brace = output.find("{")

    if left_square < left_brace and left_square != -1:
        left = left_square
        right = output.rfind("]")
    else:
        left = left_brace
        right = output.rfind("}")

    return output[left : right + 1]


def parse_json_markdown(text: str) -> Any:
    if "```json" in text:
        text = text.split("```json")[1].strip().strip("```").strip()

    json_string = _marshal_llm_to_json(text)

    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e_json:
        try:
            # NOTE: parsing again with pyyaml
            #       pyyaml is less strict, and allows for trailing commas
            #       right now we rely on this since guidance program generates
            #       trailing commas
            json_obj = yaml.safe_load(json_string)
        except yaml.YAMLError as e_yaml:
            raise OutputParserException(
                f"Got invalid JSON object. Error: {e_json} {e_yaml}. "
                f"Got JSON string: {json_string}"
            )
        except NameError as exc:
            raise ImportError("Please pip install PyYAML.") from exc

    return json_obj


def parse_code_markdown(text: str, only_last: bool) -> List[str]:
    # Regular expression pattern to match code within triple-backticks
    pattern = r"```(.*?)```"

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the last matched group if requested
    code = matches[-1] if matches and only_last else matches

    # If empty we optimistically assume the output is the code
    if not code:
        # we want to handle cases where the code may start or end with triple
        # backticks
        # we also want to handle cases where the code is surrounded by regular
        # quotes
        # we can't just remove all backticks due to JS template strings

        candidate = text.strip()

        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]

        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]

        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]

        # For triple backticks we split the handling of the start and end
        # partly because there can be cases where only one and not the other
        # is present, and partly because we don't need to be so worried
        # about it being a string in a programming language
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*", "", candidate)

        if candidate.endswith("```"):
            candidate = candidate[:-3]
        code = [candidate.strip()]

    return code


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")

    return match.group()
