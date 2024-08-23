"""Test json index."""

import json
from typing import Any, Dict, cast
from unittest.mock import patch

import pytest
from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.struct_store.json_query import (
    JSONQueryEngine,
    JSONType,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.schema import QueryBundle

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
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
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
    query_engine = JSONQueryEngine(
        json_value=json_val,
        json_schema=json_schema,
        output_processor=test_output_processor,
        verbose=True,
        synthesize_response=synthesize_response,
    )

    if call_apredict:
        task = query_engine.aquery(QueryBundle("test_nl_query"))
        response: Response = cast(Response, asyncio_run(task))
    else:
        response = cast(Response, query_engine.query(QueryBundle("test_nl_query")))

    if synthesize_response:
        assert response.response == TEST_LLM_OUTPUT
    else:
        assert response.response == json.dumps([test_json_return_value])

    metadata = cast(Dict[str, Any], response.metadata)
    assert metadata["json_path_response_str"] == TEST_LLM_OUTPUT
