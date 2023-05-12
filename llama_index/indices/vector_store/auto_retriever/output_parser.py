import json
from typing import Any

from llama_index.output_parsers.base import (
    BaseOutputParser,
    OutputParserException,
    StructuredOutput,
)
from llama_index.vector_stores.types import VectorStoreQuerySpec


def parse_json_markdown(text: str) -> Any:
    if "```json" not in text:
        raise OutputParserException(
            f"Got invalid return object. Expected markdown code snippet with JSON "
            f"object, but got:\n{text}"
        )

    json_string = text.split("```json")[1].strip().strip("```").strip()
    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Got invalid JSON object. Error: {e}")

    return json_obj


class VectorStoreQueryOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = parse_json_markdown(output)

        query_and_filters = VectorStoreQuerySpec.parse_obj(json_dict)
        return StructuredOutput(raw_output=output, parsed_output=query_and_filters)

    def format(self, prompt_template: str) -> str:
        del prompt_template
        raise NotImplementedError()
