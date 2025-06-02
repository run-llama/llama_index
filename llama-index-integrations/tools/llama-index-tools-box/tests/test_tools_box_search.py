from box_sdk_gen import BoxClient
import openai
import pytest

from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.box import BoxSearchToolSpec, BoxSearchOptions

from tests.conftest import get_testing_data


def test_box_tool_search(box_client_ccg_integration_testing: BoxClient):
    options = BoxSearchOptions()
    options.limit = 5

    box_tool = BoxSearchToolSpec(box_client_ccg_integration_testing, options=options)

    query = "invoice"
    docs = box_tool.box_search(query=query)
    assert len(docs) > 0


def test_box_tool_search_options(box_client_ccg_integration_testing: BoxClient):
    options = BoxSearchOptions(file_extensions=["pdf"])
    options.limit = 5

    box_tool = BoxSearchToolSpec(box_client_ccg_integration_testing, options=options)

    query = "sample"
    docs = box_tool.box_search(query=query)
    assert len(docs) > 0


def test_box_tool_search_agent(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    openai_api_key = test_data["openai_api_key"]

    if openai_api_key is None:
        raise pytest.skip("OpenAI API key is not provided.")

    options = BoxSearchOptions()
    options.limit = 5

    box_tool_spec = BoxSearchToolSpec(
        box_client_ccg_integration_testing, options=options
    )

    openai.api_key = openai_api_key

    agent = OpenAIAgent.from_tools(
        box_tool_spec.to_tool_list(),
        verbose=True,
    )

    answer = agent.chat("search all invoices")
    # print(answer)
    assert answer is not None
