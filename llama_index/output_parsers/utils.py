import contextlib
import json
import re
from typing import Any

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


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")

    return match.group()
