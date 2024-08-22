import pytest
from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxTextExtractToolSpec
from llama_index.agent.openai import OpenAIAgent

from tests.conftest import get_testing_data


def test_box_tool_extract(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxTextExtractToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.extract(test_data["test_ppt_id"])

    assert doc.text is not None


def test_box_tool_extract_agent(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    document_id = test_data["test_ppt_id"]
    openai_api_key = test_data["openai_api_key"]

    if openai_api_key is None:
        raise pytest.skip("OpenAI API key is not provided.")

    box_tool = BoxTextExtractToolSpec(box_client=box_client_ccg_integration_testing)

    agent = OpenAIAgent.from_tools(
        box_tool.to_tool_list(),
        verbose=True,
    )

    answer = agent.chat(f"read document {document_id}")
    # print(answer)
    assert answer is not None
