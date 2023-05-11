
from typing import cast

from llama_index.indices.vector_store.auto_retriever.output_parser import \
    VectorStoreQueryOutputParser
from llama_index.output_parsers.base import StructuredOutput
from llama_index.vector_stores.types import QueryAndMetadataFilters


def test_output_parser():
    output_str = """\
    ```json
    {
        "query": "test query str",
        "filters": [
            {
                "key": "director",
                "value": "Nolan"
            },
            {
                "key": "theme",
                "value": "sci-fi"
            }
        ]
    }
    ```
    """

    parser = VectorStoreQueryOutputParser()
    output = parser.parse(output_str)
    structured_output = cast(StructuredOutput, output)
    assert isinstance(structured_output.parsed_output, QueryAndMetadataFilters)
