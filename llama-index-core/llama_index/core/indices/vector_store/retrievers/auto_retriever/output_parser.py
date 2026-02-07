from typing import Any

from pydantic import ValidationError

from llama_index.core.output_parsers.base import OutputParserException, StructuredOutput
from llama_index.core.output_parsers.utils import parse_json_markdown
from llama_index.core.types import BaseOutputParser
from llama_index.core.vector_stores.types import VectorStoreQuerySpec


class VectorStoreQueryOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = parse_json_markdown(output)
        try:
            query_and_filters = VectorStoreQuerySpec.model_validate(json_dict)
        except ValidationError as e:
            raise OutputParserException(
                f"Failed to validate query spec. Error: {e}. Got JSON dict: {json_dict}"
            ) from e

        return StructuredOutput(raw_output=output, parsed_output=query_and_filters)

    def format(self, prompt_template: str) -> str:
        return prompt_template
