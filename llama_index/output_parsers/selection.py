import json
from dataclasses import dataclass
from typing import Any, List

from dataclasses_json import DataClassJsonMixin

from llama_index.output_parsers.base import StructuredOutput
from llama_index.output_parsers.utils import _marshal_llm_to_json
from llama_index.types import BaseOutputParser


def _escape_curly_braces(input_string: str) -> str:
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces
    escaped_string = input_string.replace("{", "{{").replace("}", "}}")
    return escaped_string


FORMAT_STR = """The output should be formatted as a JSON instance that conforms to 
the JSON schema below. 

Here is the output schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""


@dataclass
class Answer(DataClassJsonMixin):
    choice: int
    reason: str


class SelectionOutputParser(BaseOutputParser):
    REQUIRED_KEYS = {"choice", "reason"}

    def _filter_dict(self, json_dict: dict) -> dict:
        output_dict = json_dict
        for key, val in json_dict.items():
            if key in self.REQUIRED_KEYS:
                continue
            elif isinstance(val, dict):
                found = True
                for key in self.REQUIRED_KEYS:
                    if key not in val:
                        found = False
                        break
                if found:
                    output_dict = val
                    break
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        output_dict = self._filter_dict(item)

        return output_dict

    def _validate_output(self, output: List[dict]) -> List[dict]:
        output_json = []
        for json_dict in output:
            valid = True
            for key in self.REQUIRED_KEYS:
                if key not in json_dict:
                    valid = False
                    break

            if not valid:
                json_dict = self._filter_dict(json_dict)

            output_json.append(json_dict)

        return output_json

    def parse(self, output: str) -> Any:
        output = _marshal_llm_to_json(output)
        json_output = json.loads(output)
        if isinstance(json_output, dict):
            json_output = [json_output]

        json_output = self._validate_output(json_output)
        answers = [Answer.from_dict(json_dict) for json_dict in json_output]
        return StructuredOutput(raw_output=output, parsed_output=answers)

    def format(self, prompt_template: str) -> str:
        fmt = prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)
        return fmt
