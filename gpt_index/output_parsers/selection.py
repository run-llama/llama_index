from dataclasses import dataclass
import json
from typing import Any

from dataclasses_json import DataClassJsonMixin
from gpt_index.output_parsers.base import BaseOutputParser, StructuredOutput


def escape_curly_braces(input_string):
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces
    escaped_string = input_string.replace("{", "{{").replace("}", "}}")
    return escaped_string


FORMAT_STR = """The output should be formatted as a JSON instance that conforms to the JSON schema below. 


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

ANSWERS_KEY = "inds"


@dataclass
class Answer(DataClassJsonMixin):
    choice: int
    reason: str


class SelectionOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_list = json.loads(output)
        answers = []
        for json_dict in json_list:
            answers.append(Answer.from_dict(json_dict))

        return StructuredOutput(raw_output=output, parsed_output={ANSWERS_KEY: answers})

    def format(self, prompt_template: str) -> str:
        fmt = prompt_template + "\n\n" + escape_curly_braces(FORMAT_STR)
        return fmt
