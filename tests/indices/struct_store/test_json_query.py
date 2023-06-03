"""Test json index."""

from typing import Any, Dict, cast, Optional
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import asyncio

import json
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import Response
from llama_index.indices.service_context import ServiceContext

from llama_index.indices.struct_store.json_query import GPTJSONQueryEngine, JSONType

TEST_PARAMS = [
    # synthesize_response, call_apredict
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]
TEST_LLM_OUTPUT = "test_llm_output"

@pytest.fixture
def mock_json_service_ctx(mock_service_context: ServiceContext) -> ServiceContext:
    with patch.object(mock_service_context, "llm_predictor") as mock_llm_predictor:
        mock_llm_predictor.apredict = AsyncMock(return_value=(TEST_LLM_OUTPUT, ""))
        mock_llm_predictor.predict = MagicMock(return_value=(TEST_LLM_OUTPUT, ""))
        yield mock_service_context

@pytest.mark.parametrize("synthesize_response,call_apredict", TEST_PARAMS)
def test_json_query_engine(synthesize_response: bool, call_apredict: bool, mock_json_service_ctx: ServiceContext) -> None:
    """Test GPTNLJSONQueryEngine."""
    # Test on some sample data
    json_val = cast(JSONType, {})
    json_schema = cast(JSONType, {})

    test_json_return_value = "test_json_return_value"

    def test_output_processor(llm_output: str, json_value: JSONType) -> JSONType:
        assert llm_output == TEST_LLM_OUTPUT
        assert json_value == json_val
        return [test_json_return_value]

    # the mock prompt just takes the first item in the given column
    query_engine = GPTJSONQueryEngine(
        json_value=json_val, json_schema=json_schema,
        service_context=mock_json_service_ctx,
        output_processor=test_output_processor,
        verbose=True,
        synthesize_response=synthesize_response,
    )

    if call_apredict:
        task = query_engine.aquery(QueryBundle("test_nl_query"))
        response: Response = cast(Response, asyncio.run(task))
    else:
        response = cast(Response, query_engine.query(QueryBundle("test_nl_query")))

    if synthesize_response:
        assert response.response == TEST_LLM_OUTPUT
    else:
        assert response.response == json.dumps([test_json_return_value])

    extra_info = cast(Dict[str, Any], response.extra_info)
    assert extra_info["json_path_response_str"] == TEST_LLM_OUTPUT
