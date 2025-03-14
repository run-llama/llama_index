import pytest
import os

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.agent import FunctionCallingAgent

from llama_index.tools.agentql import AgentQLRestAPIToolSpec
from llama_index.llms.openai import OpenAI

from llama_index.tools.agentql.test_data import TEST_DATA, TEST_URL, TEST_QUERY


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentQLRestAPIToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


class TestExtractDataRestApiTool:
    @pytest.fixture()
    def agent(self):
        agentql_rest_api_tool = AgentQLRestAPIToolSpec()
        return FunctionCallingAgent.from_tools(
            agentql_rest_api_tool.to_tool_list(),
            llm=OpenAI(model="gpt-4o"),
        )

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ or "AGENTQL_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY or AGENTQL_API_KEY is not set",
    )
    def test_extract_web_data_llm_tool_call(self, agent):
        res = agent.chat(
            f"""
        extract the data from {TEST_URL} with the following agentql query: {TEST_QUERY}
        """
        )
        tool_output = res.sources[0]
        assert tool_output.tool_name == "extract_web_data_with_rest_api"
        assert tool_output.raw_input["kwargs"] == {
            "url": TEST_URL,
            "query": TEST_QUERY,
        }
        assert tool_output.raw_output["data"] == TEST_DATA
