
import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_index.core.query_engine.sql_join_query_engine import SQLJoinQueryEngine
from llama_index.core.service_context import ServiceContext
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.indices.struct_store.sql_query import BaseSQLTableQueryEngine
from llama_index.core.schema import QueryBundle, Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.core.query_engine.sql_join_query_engine import SQLAugmentQueryTransform

class MockSQLQueryEngine(BaseSQLTableQueryEngine):
    def __init__(self):
        # Initialize required abstract methods or properties if any
        pass
        
    def query(self, query_bundle):
        return Response(response="SQL Response", metadata={"sql_query": "SELECT * FROM table"})
        
    async def aquery(self, query_bundle):
        return Response(response="SQL Response", metadata={"sql_query": "SELECT * FROM table"})

@pytest.fixture
def mock_sql_tool():
    query_engine = MagicMock(spec=BaseSQLTableQueryEngine)
    query_engine.query.return_value = Response(
        response="SQL Response", 
        metadata={"sql_query": "SELECT * FROM table"}
    )
    query_engine.aquery = AsyncMock(return_value=Response(
        response="SQL Response", 
        metadata={"sql_query": "SELECT * FROM table"}
    ))
    # We need to simulate the metadata attribute of the tool
    tool = QueryEngineTool.from_defaults(query_engine=query_engine, name="sql_tool", description="SQL Tool")
    return tool

@pytest.fixture
def mock_other_tool():
    query_engine = MagicMock()
    query_engine.query.return_value = Response(response="Other Response")
    query_engine.aquery = AsyncMock(return_value=Response(response="Other Response"))
    tool = QueryEngineTool.from_defaults(query_engine=query_engine, name="other_tool", description="Other Tool")
    return tool

@pytest.fixture
def mock_info_selector():
    selector = MagicMock(spec=LLMSingleSelector)
    return selector

@pytest.fixture
def mock_llm():
    return MockLLM()

def test_sql_join_query_engine_sync(mock_sql_tool, mock_other_tool, mock_llm):
    # Setup selector to pick SQL tool (index 0)
    selector = MagicMock(spec=LLMSingleSelector)
    selector.select.return_value.ind = 0
    selector.select.return_value.reason = "test reason"
    
    engine = SQLJoinQueryEngine(
        sql_query_tool=mock_sql_tool,
        other_query_tool=mock_other_tool,
        selector=selector,
        llm=mock_llm,
        use_sql_join_synthesis=False # Simplify for basic test
    )
    
    response = engine.query(QueryBundle("test query"))
    assert str(response) == "SQL Response"
    mock_sql_tool.query_engine.query.assert_called_once()


@pytest.mark.asyncio
async def test_sql_join_query_engine_async(mock_sql_tool, mock_other_tool, mock_llm):
    # Setup selector to pick SQL tool (index 0)
    selector = MagicMock(spec=LLMSingleSelector)
    selector.aselect = AsyncMock()
    selector.aselect.return_value.ind = 0
    selector.aselect.return_value.reason = "test reason"

    # Fallback to synchronous select if aselect not present (?) 
    # But we want to test true async if possible. 
    # SQLJoinQueryEngine._selector type constraint includes PydanticSingleSelector which has aselect.
    
    engine = SQLJoinQueryEngine(
        sql_query_tool=mock_sql_tool,
        other_query_tool=mock_other_tool,
        selector=selector,
        llm=mock_llm,
        use_sql_join_synthesis=False
    )
    
    response = await engine.aquery(QueryBundle("test query"))
    assert str(response) == "SQL Response"
    
    # helper for checking calls
    assert mock_sql_tool.query_engine.aquery.call_count == 1
