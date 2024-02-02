"""Test json index."""

import asyncio
import json
from typing import Any, Dict, cast
from unittest.mock import patch

import pytest
from llama_index.legacy.core.response.schema import Response
from llama_index.legacy.indices.struct_store.json_query import (
    JSONQueryEngine,
    JSONType,
)
from llama_index.legacy.llm_predictor import LLMPredictor
from llama_index.legacy.llms.mock import MockLLM
from llama_index.legacy.prompts.base import BasePromptTemplate
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.service_context import ServiceContext

TEST_PARAMS = [
    # synthesize_response, call_apredict
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]
TEST_LLM_OUTPUT = "test_llm_output"


def mock_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
    return TEST_LLM_OUTPUT


async def amock_predict(
    self: Any, prompt: BasePromptTemplate, **prompt_args: Any
) -> str:
    return TEST_LLM_OUTPUT


@pytest.mark.parametrize(("synthesize_response", "call_apredict"), TEST_PARAMS)
@patch.object(
    MockLLM,
    "predict",
    mock_predict,
)
@patch.object(
    MockLLM,
    "apredict",
    amock_predict,
)
def test_json_query_engine(
    synthesize_response: bool,
    call_apredict: bool,
    mock_service_context: ServiceContext,
) -> None:
    """Test GPTNLJSONQueryEngine."""
    mock_service_context.llm_predictor = LLMPredictor(MockLLM())

    # Test on some sample data
    json_val = cast(JSONType, {})
    json_schema = cast(JSONType, {})

    test_json_return_value = "test_json_return_value"

    def test_output_processor(llm_output: str, json_value: JSONType) -> JSONType:
        assert llm_output == TEST_LLM_OUTPUT
        assert json_value == json_val
        return [test_json_return_value]

    # the mock prompt just takes the first item in the given column
    query_engine = JSONQueryEngine(
        json_value=json_val,
        json_schema=json_schema,
        service_context=mock_service_context,
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

    metadata = cast(Dict[str, Any], response.metadata)
    assert metadata["json_path_response_str"] == TEST_LLM_OUTPUT
