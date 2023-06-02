"""Test json index."""

from typing import Any, Dict, cast
from unittest.mock import MagicMock

import json
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext

from llama_index.indices.struct_store.json import GPTJSONIndex, JSONType
from llama_index.indices.struct_store.json_query import GPTNLJSONQueryEngine


def test_json_index(mock_service_context: ServiceContext) -> None:
    """Test GPTJSONIndex."""
    # Test on some sample data
    json_val = {}
    json_schema = {}
    index = GPTJSONIndex(
        json_value=json_val,
        json_schema=json_schema,
        service_context=mock_service_context,
    )

    test_llm_output = "test_llm_output"
    mock_service_context.llm_predictor.predict = MagicMock(
        return_value=(test_llm_output, "")
    )
    test_json_return_value = "test_json_return_value"

    def test_output_processor(llm_output, json_value: JSONType) -> JSONType:
        assert llm_output == test_llm_output
        assert json_value == json_val
        return [test_json_return_value]

    # the mock prompt just takes the first item in the given column
    query_engine = GPTNLJSONQueryEngine(
        index=index, output_processor=test_output_processor, verbose=True
    )
    response = query_engine.query(QueryBundle("test_nl_query"))

    assert response.response == json.dumps([test_json_return_value])

    extra_info = cast(Dict[str, Any], response.extra_info)
    assert extra_info["json_path_response_str"] == test_llm_output
