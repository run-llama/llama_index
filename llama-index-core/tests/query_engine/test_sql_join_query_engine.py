import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_index.core.query_engine.sql_join_query_engine import SQLJoinQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.indices.struct_store.sql_query import BaseSQLTableQueryEngine
from llama_index.core.schema import QueryBundle, Response
from llama_index.core.llms.mock import MockLLM


class MockSQLQueryEngine(BaseSQLTableQueryEngine):
    def __init__(self):
        pass

    def query(self, q):
        return Response(response="SQL Response", metadata={"sql_query": "SELECT *"})

    async def aquery(self, q):
        return Response(response="SQL Response", metadata={"sql_query": "SELECT *"})


@pytest.fixture
def mock_sql_tool():
    qe = MagicMock(spec=BaseSQLTableQueryEngine)
    qe.query.return_value = Response(
        response="SQL Response", metadata={"sql_query": "SELECT *"}
    )
    qe.aquery = AsyncMock(
        return_value=Response(
            response="SQL Response", metadata={"sql_query": "SELECT *"}
        )
    )
    return QueryEngineTool.from_defaults(
        query_engine=qe, name="sql_tool", description="SQL Tool"
    )


@pytest.fixture
def mock_other_tool():
    qe = MagicMock()
    qe.query.return_value = Response(response="Other Response")
    qe.aquery = AsyncMock(return_value=Response(response="Other Response"))
    return QueryEngineTool.from_defaults(
        query_engine=qe, name="other_tool", description="Other Tool"
    )


@pytest.fixture
def mock_info_selector():
    return MagicMock(spec=LLMSingleSelector)


@pytest.fixture
def mock_llm():
    return MockLLM()


def test_sql_join_query_engine_sync(mock_sql_tool, mock_other_tool, mock_llm):
    s = MagicMock(spec=LLMSingleSelector)
    s.select.return_value.ind = 0
    s.select.return_value.reason = "r"
    e = SQLJoinQueryEngine(
        sql_query_tool=mock_sql_tool,
        other_query_tool=mock_other_tool,
        selector=s,
        llm=mock_llm,
        use_sql_join_synthesis=False,
    )
    assert str(e.query(QueryBundle("q"))) == "SQL Response"


@pytest.mark.asyncio
async def test_sql_join_query_engine_async(mock_sql_tool, mock_other_tool, mock_llm):
    s = MagicMock(spec=LLMSingleSelector)
    s.aselect = AsyncMock()
    s.aselect.return_value.ind = 0
    s.aselect.return_value.reason = "r"
    e = SQLJoinQueryEngine(
        sql_query_tool=mock_sql_tool,
        other_query_tool=mock_other_tool,
        selector=s,
        llm=mock_llm,
        use_sql_join_synthesis=False,
    )
    assert str(await e.aquery(QueryBundle("q"))) == "SQL Response"
    assert mock_sql_tool.query_engine.aquery.call_count == 1
