from typing import cast

import pytest

from llama_index.core.indices.vector_store.retrievers.auto_retriever.output_parser import (
    VectorStoreQueryOutputParser,
)
from llama_index.core.output_parsers.base import OutputParserException, StructuredOutput
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    VectorStoreQuerySpec,
)


def test_output_parser() -> None:
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
        ],
        "top_k": 2
    }
    ```
    """

    parser = VectorStoreQueryOutputParser()
    output = parser.parse(output_str)
    structured_output = cast(StructuredOutput, output)
    assert isinstance(structured_output.parsed_output, VectorStoreQuerySpec)

    expected = VectorStoreQuerySpec(
        query="test query str",
        filters=[
            ExactMatchFilter(key="director", value="Nolan"),
            ExactMatchFilter(key="theme", value="sci-fi"),
        ],
        top_k=2,
    )
    assert structured_output.parsed_output == expected


def test_output_parser_invalid_schema_raises_output_parser_exception() -> None:
    """Test that invalid JSON schema raises OutputParserException instead of ValidationError."""
    # JSON with invalid filter structure (missing required 'key' field)
    output_str = """\
    ```json
    {
        "query": "test query",
        "filters": [
            {
                "value": "missing key field"
            }
        ],
        "top_k": 5
    }
    ```
    """

    parser = VectorStoreQueryOutputParser()
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(output_str)

    assert "Failed to validate query spec" in str(exc_info.value)


def test_output_parser_invalid_type_raises_output_parser_exception() -> None:
    """Test that invalid type for top_k raises OutputParserException."""
    # JSON with invalid type (top_k should be int, not string)
    output_str = """\
    ```json
    {
        "query": "test query",
        "filters": [],
        "top_k": "not_a_number"
    }
    ```
    """

    parser = VectorStoreQueryOutputParser()
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(output_str)

    assert "Failed to validate query spec" in str(exc_info.value)


def test_output_parser_keeps_boolean_filter() -> None:
    """A boolean filter value must not make the whole query spec invalid."""
    output_str = """\
    ```json
    {
        "query": "articles",
        "filters": [
            {"key": "year", "value": 2024},
            {"key": "is_published", "value": true}
        ],
        "top_k": null
    }
    ```
    """

    output = cast(StructuredOutput, VectorStoreQueryOutputParser().parse(output_str))

    assert output.parsed_output.filters == [
        ExactMatchFilter(key="year", value=2024),
        ExactMatchFilter(key="is_published", value=True),
    ]
