import openai
import pytest
from box_sdk_gen import BoxClient
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.box import BoxAIExtractToolSpec

from tests.conftest import get_testing_data

AI_PROMPT = (
    '{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}'
)


def test_box_tool_ai_extract(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxAIExtractToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.ai_extract(
        file_id=test_data["test_txt_invoice_id"], ai_prompt=AI_PROMPT
    )

    assert doc.text is not None


def test_box_tool_ai_extract_agent(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    document_id = test_data["test_txt_invoice_id"]
    openai_api_key = test_data["openai_api_key"]
    ai_prompt = (
        '{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}'
    )

    if openai_api_key is None:
        raise pytest.skip("OpenAI API key is not provided.")

    openai.api_key = openai_api_key

    box_tool = BoxAIExtractToolSpec(box_client=box_client_ccg_integration_testing)

    agent = OpenAIAgent.from_tools(
        box_tool.to_tool_list(),
        verbose=True,
    )

    answer = agent.chat(f"{ai_prompt} for {document_id}")
    # print(answer)
    assert answer is not None
