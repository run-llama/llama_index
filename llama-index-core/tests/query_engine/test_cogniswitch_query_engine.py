import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine.cogniswitch_query_engine import (
    CogniswitchQueryEngine,
)
from llama_index.core.schema import QueryBundle


@pytest.fixture()
def query_engine() -> CogniswitchQueryEngine:
    return CogniswitchQueryEngine(
        cs_token="cs_token", OAI_token="OAI_token", apiKey="api_key"
    )


@patch("requests.post")
def test_query_knowledge_successful(
    mock_post: Any, query_engine: CogniswitchQueryEngine
) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"data": {"answer": "42"}}
    response = query_engine.query_knowledge("What is the meaning of life?")
    assert isinstance(response, Response)
    assert response.response == "42"


@patch("requests.post")
def test_query_knowledge_unsuccessful(
    mock_post: Any, query_engine: CogniswitchQueryEngine
) -> None:
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {"message": "Bad Request"}
    response = query_engine.query_knowledge("what is life?")
    assert isinstance(response, Response)
    assert response.response == "Bad Request"


@pytest.mark.asyncio
@patch("requests.post")
async def test_aquery_offloads_blocking_call_to_thread(
    mock_post: Any, query_engine: CogniswitchQueryEngine
) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"data": {"answer": "42"}}

    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy:
        response = await query_engine._aquery(
            QueryBundle(query_str="What is the meaning of life?")
        )

    # The blocking requests.post must run off the event loop via asyncio.to_thread,
    # while the response is unchanged.
    spy.assert_called_once()
    assert isinstance(response, Response)
    assert response.response == "42"
