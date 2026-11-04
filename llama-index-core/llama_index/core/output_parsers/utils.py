import contextlib
import json
import re
from typing import Any, List

with contextlib.suppress(ImportError):
    import yaml

from llama_index.core.output_parsers.base import OutputParserException


def _match_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
    """Return the index of the bracket that closes the one at ``start``.

    Brackets inside JSON strings are ignored. Returns ``-1`` when the bracket
    is never closed.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _looks_like_json_payload(candidate: str) -> bool:
    """Whether ``candidate`` parses to a JSON object or an array of objects."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    if isinstance(parsed, dict):
        return True
    return isinstance(parsed, list) and any(isinstance(item, dict) for item in parsed)


def _marshal_llm_to_json(output: str) -> str:
    """
    Extract a substring containing valid JSON or array from a string.

    Args:
        output: A string that may contain a valid JSON object or array surrounded by
        extraneous characters or information.

    Returns:
        A string containing a valid JSON object or array.

    """
    output = output.strip()

    left_square = output.find("[")
    left_brace = output.find("{")

    if left_square == -1 and left_brace == -1:
        return output

    if left_square != -1 and (left_brace == -1 or left_square < left_brace):
        end = _match_bracket(output, left_square, "[", "]")
        candidate = output[left_square : end + 1] if end != -1 else ""
        # A leading bracket group that is not a JSON payload (an inline citation
        # such as "[1]" or "[Model-A]") is prose; keep looking for a JSON object.
        if not candidate or left_brace == -1 or _looks_like_json_payload(candidate):
            return candidate or output[left_square : output.rfind("]") + 1]

    end = _match_bracket(output, left_brace, "{", "}")
    return (
        output[left_brace : end + 1]
        if end != -1
        else output[left_brace : output.rfind("}") + 1]
    )


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

    # Regular expression pattern to match code within triple backticks with
    # a Python marker. Like: ```python df.columns```
    python_str_pattern = re.compile(r"^```python", re.IGNORECASE)
    text = python_str_pattern.sub("```", text)

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
