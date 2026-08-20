import json
from dataclasses import dataclass
from typing import Any, List

from dataclasses_json import DataClassJsonMixin
from llama_index.core.output_parsers.base import (
    OutputParserException,
    StructuredOutput,
)
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.types import BaseOutputParser


def _escape_curly_braces(input_string: str) -> str:
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces
    return input_string.replace("{", "{{").replace("}", "}}")


FORMAT_STR = """The output should be ONLY JSON formatted as a JSON instance.

Here is an example:
[
    {
        choice: 1,
        reason: "<insert reason for choice>"
    },
    ...
]
"""


@dataclass
class Answer(DataClassJsonMixin):
    choice: int
    reason: str


class SelectionOutputParser(BaseOutputParser):
    REQUIRED_KEYS = frozenset(Answer.__annotations__)

    def _filter_dict(self, json_dict: dict) -> List[dict]:
        """
        Recursively collect every nested dict that matches all REQUIRED_KEYS.

        A model may wrap the answers in an outer object (e.g. ``{"items": [...]}``), and
        that list can hold more than one selection, so all matches are returned rather than
        just the last one encountered.
        """
        if self.REQUIRED_KEYS.issubset(json_dict):
            return [json_dict]

        matches: List[dict] = []
        for val in json_dict.values():
            if isinstance(val, dict):
                matches.extend(self._filter_dict(val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        matches.extend(self._filter_dict(item))

        return matches

    def _format_output(self, output: List[dict]) -> List[dict]:
        output_json = []
        for json_dict in output:
            if self.REQUIRED_KEYS.issubset(json_dict):
                output_json.append(json_dict)
                continue

            # Dig the selections out of a wrapper object; keep the original dict when
            # nothing matches so downstream parsing still surfaces the error.
            matches = self._filter_dict(json_dict)
            output_json.extend(matches or [json_dict])

        return output_json

    def parse(self, output: str) -> Any:
        json_string = _marshal_llm_to_json(output)
        try:
            json_obj = json.loads(json_string)
        except json.JSONDecodeError as e_json:
            try:
                import yaml

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

        if isinstance(json_obj, dict):
            json_obj = [json_obj]

        if not isinstance(json_obj, list):
            raise ValueError(f"Failed to convert output to JSON: {output!r}")

        json_output = self._format_output(json_obj)
        answers = [Answer.from_dict(json_dict) for json_dict in json_output]
        return StructuredOutput(raw_output=output, parsed_output=answers)

    def format(self, prompt_template: str) -> str:
        return prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)
