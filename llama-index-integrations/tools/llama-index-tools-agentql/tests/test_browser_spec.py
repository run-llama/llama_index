import os

import pytest
from llama_index.core.agent.function_calling.base import FunctionCallingAgent
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.tools.agentql import AgentQLBrowserToolSpec
from llama_index.tools.playwright import PlaywrightToolSpec

from tests.conftest import get_testing_data


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentQLBrowserToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


class TestExtractDataBrowserTool:
    @pytest.fixture(autouse=True)
    async def agentql_browser_tool(self):
        test_data = get_testing_data()
        # Use playwright tool to navigate to the test url
        async_browser = await PlaywrightToolSpec.create_async_playwright_browser()
        playwright_tool = PlaywrightToolSpec.from_async_browser(async_browser)
        await playwright_tool.navigate_to(test_data["TEST_URL"])

        # initialize extract data browser tool
        agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)
        yield agentql_browser_tool
        await async_browser.close()

    @pytest.fixture
    def agent(self, agentql_browser_tool):
        return FunctionCallingAgent.from_tools(
            agentql_browser_tool.to_tool_list(),
            llm=OpenAI(model="gpt-4o"),
        )

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ or "AGENTQL_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY or AGENTQL_API_KEY is not set",
    )
    def test_extract_web_data_browser_tool_call(self, agent):
        test_data = get_testing_data()
        res = agent.chat(
            f"""
        extract data with the following agentql query: {test_data["TEST_QUERY"]}
        """
        )
        tool_output = res.sources[0]
        assert tool_output.tool_name == "extract_web_data_from_browser"
        assert tool_output.raw_input["kwargs"] == {
            "query": test_data["TEST_QUERY"],
        }
        assert tool_output.raw_output == test_data["TEST_DATA"]

    @pytest.mark.skipif(
        "AGENTQL_API_KEY" not in os.environ,
        reason="AGENTQL_API_KEY is not set",
    )
    async def test_get_web_element_browser_tool_call(self, agentql_browser_tool):
        next_page_button = await agentql_browser_tool.get_web_element_from_browser(
            prompt="button for buying it now",
        )
        assert next_page_button == "[tf623_id='965']"
