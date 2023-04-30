import re
from typing import Any, List
from gpt_index.output_parsers.base import BaseOutputParser, StructuredOutput


FORMAT_STR = (
    "Provide answers in JSON format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)

INDS_KEY = "inds"


def extract_numbers(response: str) -> List[int]:
    """Extract number given the GPT-generated response."""
    outputs = re.findall(r"\d+", response)
    return [int(output) for output in outputs]


class SelectionOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        inds = extract_numbers(output)
        return StructuredOutput(raw_output=output, parsed_output={INDS_KEY: inds})

    def format(self, prompt_template: str) -> str:
        fmt = prompt_template + "\n\n" + FORMAT_STR
        return fmt
