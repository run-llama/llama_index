import json
from dataclasses import dataclass
from typing import Any

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
    def parse(self, output: str) -> Any:
        output = _marshal_llm_to_json(output)
        json_output = json.loads(output)
        if isinstance(json_output, dict):
            json_output = [json_output]
        answers = [Answer.from_dict(json_dict) for json_dict in json_output]
        return StructuredOutput(raw_output=output, parsed_output=answers)

    def format(self, prompt_template: str) -> str:
        fmt = prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)
        return fmt
