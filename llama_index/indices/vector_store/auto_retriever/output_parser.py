


import json
from typing import Any

from llama_index.output_parsers.base import BaseOutputParser, StructuredOutput
from llama_index.vector_stores.types import QueryAndMetadataFilters


class VectorStoreQueryOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = json.loads(output)

        query_and_filters = QueryAndMetadataFilters.from_dict(json_dict)
        return StructuredOutput(raw_output=output, parsed_output=query_and_filters)
    
    def format(self, prompt_template: str) -> str:
        raise NotImplementedError()
