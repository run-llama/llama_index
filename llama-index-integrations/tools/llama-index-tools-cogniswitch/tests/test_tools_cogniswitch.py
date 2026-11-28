import asyncio
from unittest.mock import patch

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.cogniswitch import (
    CogniswitchQueryEngine,
    CogniswitchToolSpec,
)


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in CogniswitchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.mark.asyncio
async def test_cogniswitch_query_engine_aquery_offloads_to_thread() -> None:
    engine = CogniswitchQueryEngine(
        cs_token="test_cs", OAI_token="test_oai", apiKey="test_key"
    )
    expected_response = Response(response="mocked answer")

    with patch.object(
        asyncio, "to_thread", wraps=asyncio.to_thread
    ) as mock_to_thread, patch.object(
        engine, "query_knowledge", return_value=expected_response
    ) as mock_query_knowledge:
        result = await engine._aquery(QueryBundle(query_str="hello"))

    assert result == expected_response
    mock_to_thread.assert_called_once_with(engine.query_knowledge, "hello")
    mock_query_knowledge.assert_called_once_with("hello")
