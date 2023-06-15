import json
from typing import Any

import yaml

from llama_index.output_parsers.base import OutputParserException


def parse_json_markdown(text: str) -> Any:
    if "```json" not in text:
        raise OutputParserException(
            f"Got invalid return object. Expected markdown code snippet with JSON "
            f"object, but got:\n{text}"
        )

    json_string = text.split("```json")[1].strip().strip("```").strip()
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
                f"Got invalid JSON object. Error: {e_json} {e_yaml}"
            )

    return json_obj
